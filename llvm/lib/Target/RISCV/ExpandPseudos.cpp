//===-- ExpandPseudos.cpp - Expand Pseudos ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpandPseudos.h"
#include <climits>
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

#define DEBUG_TYPE "expandpseudos"
#define RISCV_INSERT_VSETVLI_NAME "Custom Expand to Pseudos and Sink pass"

char ExpandPseudos::ID = 0;
char &llvm::ExpandPseudosID = ExpandPseudos::ID;

INITIALIZE_PASS(ExpandPseudos, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static cl::opt<bool>
    Sink("custom-sink", cl::init(false), cl::Hidden,
               cl::desc("Enable sinking"));

static cl::opt<bool>
    Remat("custom-remat", cl::init(false), cl::Hidden,
          cl::desc("Turn redundant masked reloads into full loads and reload "
                   "whole register loads before distant uses"));

// Minimum distance, in instructions, between two uses of a loaded value for
// the later use to get its own reload, and the number of uses required before
// that gap.
static constexpr unsigned RematGap = 6;
static constexpr unsigned RematMinUses = 2;

static cl::opt<bool>
    a("custom-a", cl::init(false), cl::Hidden,
               cl::desc("Enable affine"));

static cl::opt<bool>
    Reverse("custom-reverse", cl::init(false), cl::Hidden,
            cl::desc("Reverse rematerialize values used early and needed "
                     "again much later (possibly chained over several "
                     "invertible ops) instead of keeping them live the "
                     "whole time in between"));

static cl::opt<bool>
    Forward("custom-forward", cl::init(false), cl::Hidden,
            cl::desc("Forward rematerialize values used early and needed "
                     "again much later, by replaying the original chain of "
                     "ops from a root that is already live for the whole "
                     "region anyway (paper's Figure 2(b)); needs no "
                     "algebraic inverse, so it also covers non-invertible "
                     "ops like shifts and bitwise ops that -custom-reverse "
                     "cannot"));

ExpandPseudos::ExpandPseudos() : MachineFunctionPass(ID) {
  initializeExpandPseudosPass(*PassRegistry::getPassRegistry());
}

void ExpandPseudos::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervalsWrapperPass>();
  //AU.addRequired<MachineDominatorTreeWrapperPass>();
  //AU.addRequired<MachineCycleInfoWrapperPass>();
  //AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static bool isVectorReg(Register R, const MachineRegisterInfo &MRI) {
  if (R.isVirtual())
    return RISCVRI::isVRegClass(MRI.getRegClass(R)->TSFlags);
  return R.isPhysical() && RISCV::VRRegClass.contains(R);
}

// Sum of live interval lengths (SLIL) of the vector virtual registers used in
// the block, copied from RISCVRegisterPressure.cpp. The last use of a register
// does not count towards register pressure, so the length of a register is
// (Last - First) * LMUL, where First and Last are the first and last
// instruction referencing it, counted in instructions that touch a vector
// register. Used to check the incremental update of the lengths.
static unsigned computeSLIL(const MachineBasicBlock &MBB,
                            const MachineRegisterInfo &MRI,
                            const TargetRegisterInfo &TRI) {
  DenseMap<Register, std::pair<unsigned, unsigned>> Range;
  unsigned Row = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    bool Touches = false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg() || !isVectorReg(MO.getReg(), MRI))
        continue;
      Touches = true;
      if (MO.getReg().isVirtual()) {
        auto It = Range.try_emplace(MO.getReg(), Row, Row).first;
        It->second.second = Row;
      }
    }
    if (Touches)
      ++Row;
  }
  unsigned Sum = 0;
  for (auto &KV : Range) {
    unsigned LMUL = TRI.getRegClassWeight(MRI.getRegClass(KV.first)).RegWeight;
    Sum += (KV.second.second - KV.second.first) * LMUL;
  }
  return Sum;
}

// Matches whole register loads named VL<NF>RE<EEW>_V, e.g. VL4RE32_V.
static bool isWholeRegLoadName(StringRef Name) {
  if (!Name.consume_front("VL") || !Name.consume_back("_V"))
    return false;
  StringRef NF, EEW;
  std::tie(NF, EEW) = Name.split("RE");
  auto IsNumber = [](StringRef S) {
    return !S.empty() && llvm::all_of(S, isDigit);
  };
  return IsNumber(NF) && IsNumber(EEW);
}

static bool attemptDebugCopyProp(MachineInstr &SinkInst, MachineInstr &DbgMI,
                                 Register Reg) {
  const MachineRegisterInfo &MRI = SinkInst.getMF()->getRegInfo();
  const TargetInstrInfo &TII = *SinkInst.getMF()->getSubtarget().getInstrInfo();

  // Copy DBG_VALUE operand and set the original to undef. We then check to
  // see whether this is something that can be copy-forwarded. If it isn't,
  // continue around the loop.

  const MachineOperand *SrcMO = nullptr, *DstMO = nullptr;
  auto CopyOperands = TII.isCopyInstr(SinkInst);
  if (!CopyOperands)
    return false;
  SrcMO = CopyOperands->Source;
  DstMO = CopyOperands->Destination;

  // Check validity of forwarding this copy.
  bool PostRA = MRI.getNumVirtRegs() == 0;

  // Trying to forward between physical and virtual registers is too hard.
  if (Reg.isVirtual() != SrcMO->getReg().isVirtual())
    return false;

  // Only try virtual register copy-forwarding before regalloc, and physical
  // register copy-forwarding after regalloc.
  bool arePhysRegs = !Reg.isVirtual();
  if (arePhysRegs != PostRA)
    return false;

  // Pre-regalloc, only forward if all subregisters agree (or there are no
  // subregs at all). More analysis might recover some forwardable copies.
  if (!PostRA)
    for (auto &DbgMO : DbgMI.getDebugOperandsForReg(Reg))
      if (DbgMO.getSubReg() != SrcMO->getSubReg() ||
          DbgMO.getSubReg() != DstMO->getSubReg())
        return false;

  // Post-regalloc, we may be sinking a DBG_VALUE of a sub or super-register
  // of this copy. Only forward the copy if the DBG_VALUE operand exactly
  // matches the copy destination.
  if (PostRA && Reg != DstMO->getReg())
    return false;

  for (auto &DbgMO : DbgMI.getDebugOperandsForReg(Reg)) {
    DbgMO.setReg(SrcMO->getReg());
    DbgMO.setSubReg(SrcMO->getSubReg());
  }
  return true;
}

bool ExpandPseudos::LowerCopy(MachineBasicBlock &MBB, MachineInstr &MI) {
  bool MadeChange = false;
  if (MI.getNumOperands() >= 2 &&
            MI.getOperand(0).isReg() &&
            MI.getOperand(1).isReg()) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
    if (DstReg.isVirtual() && SrcReg.isVirtual()) {
      const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
      const TargetRegisterClass *SrcRC = MRI->getRegClass(SrcReg);
      StringRef DstName = TRI->getRegClassName(DstRC);
      StringRef SrcName = TRI->getRegClassName(SrcRC);
      if (DstName.contains("VRM8") && SrcName.contains("VRM8")) {
        LLVM_DEBUG(dbgs() << "lowerCopy: "<<MI);
        unsigned SEW = 5;
        DebugLoc DL = MI.getDebugLoc();
        MachineBasicBlock::iterator MBBI = MI.getIterator();
        Register VLReg;
        Register VXReg;
        unsigned NewOpCode = 0;
        bool isVI = false;
        auto CreateNewVMV = [&](unsigned OpCode, bool isVI,
                                MachineInstr &PrevMI) -> MachineInstr * {
          MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL,
                                            TII->get(OpCode));
          MIB.addReg(DstReg, RegState::Define);
          MIB.addReg(DstReg, RegState::Undef);
          if (isVI)
            MIB.addImm(0);
          else
            MIB.addReg(VXReg);
          MIB.addReg(VLReg);
          MIB.addImm(SEW);
          MIB.addImm(0);
          MIB.copyImplicitOps(PrevMI);
          return MIB;
        };
        for (auto It = MBB.begin(); It != MI.getIterator(); ++It) {
          MachineInstr &PrevMI = *It;
          if (PrevMI.getOpcode() == RISCV::PseudoVMV_V_I_M8 &&
            PrevMI.getOperand(0).isReg() &&
            PrevMI.getOperand(0).getReg() == SrcReg &&
            PrevMI.getNumOperands() >= 2 &&
            PrevMI.getOperand(2).isImm() &&
            PrevMI.getOperand(2).getImm() == 0) {
            if (PrevMI.getNumOperands() >= 4) {
              if (PrevMI.getOperand(3).isReg())
                 VLReg = PrevMI.getOperand(3).getReg();
            }
            NewOpCode = RISCV::PseudoVMV_V_I_M8;
            isVI = true;
          }
          else if (PrevMI.getOpcode() == RISCV::PseudoVMV_V_X_M8 &&
            PrevMI.getOperand(0).isReg() &&
            PrevMI.getOperand(0).getReg() == SrcReg &&
            PrevMI.getNumOperands() >= 2 &&
            PrevMI.getOperand(2).isReg()) {
            if (PrevMI.getNumOperands() >= 4) {
              if (PrevMI.getOperand(2).isReg())
                 VXReg = PrevMI.getOperand(2).getReg();
              if (PrevMI.getOperand(3).isReg())
                 VLReg = PrevMI.getOperand(3).getReg();
            }
            NewOpCode = RISCV::PseudoVMV_V_X_M8;
            isVI = false;
          }
          if (NewOpCode) {
            LLVM_DEBUG(dbgs() << "Replacing COPY with VMV: SEW=" << SEW << "\n");
            auto NewMI = CreateNewVMV(NewOpCode, isVI, PrevMI);
            NewMI->setDebugLoc(DL);
            MI.eraseFromParent();
            // NewMI was inserted, and MI erased, without updating
            // SlotIndexes/LiveIntervals: LIS no longer matches the
            // instruction list, so later code must not call into it.
            LISValid = false;
            return true;
          }
        }
      }
    }
  }
  return MadeChange;
}

// Locate the first non-debug use of Reg after Def in Def's block. The uses come
// from the register's use list, so the walk only tests set membership instead
// of scanning the operands of every instruction. Returns null if some use is in
// another block, or there is none. Between, if given, is the number of
// non-debug instructions between Def and the first use.
static MachineInstr *findFirstUseInBlock(MachineInstr &Def, Register Reg,
                                         MachineRegisterInfo &MRI,
                                         unsigned *Between = nullptr) {
  MachineBasicBlock *MBB = Def.getParent();
  SmallPtrSet<const MachineInstr *, 8> Uses;
  for (MachineInstr &U : MRI.use_nodbg_instructions(Reg)) {
    if (U.getParent() != MBB)
      return nullptr;
    if (&U != &Def)
      Uses.insert(&U);
  }
  unsigned Count = 0;
  for (auto It = std::next(Def.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    if (Uses.count(&*It)) {
      if (Between)
        *Between = Count;
      return &*It;
    }
    ++Count;
  }
  return nullptr;
}

// Sink MI down to just before the first use of its result:
//   def A
//   ...
//   C = A * 2   -->   ... ; def A ; C = A * 2
bool ExpandPseudos::FindFirstUseToSinkTo(
    MachineInstr &MI, AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  FindFirstUseToSinkTo.\n");
  if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg())
    return false;
  Register Reg = MI.getOperand(0).getReg();
  if (!Reg.isVirtual())
    return false;
  MachineBasicBlock *MBB = MI.getParent();
  bool SawStore = false;
  if (!MI.isSafeToMove(SawStore) || MI.isConvergent()) {
    LLVM_DEBUG(dbgs() << "  Not safe to move.\n");
    return false;
  }

  // Find the first use in this block. Uses elsewhere keep the value live past
  // the block, so leave those alone.
  unsigned N = 0;
  MachineInstr *FirstUse = findFirstUseInBlock(MI, Reg, *MRI, &N);
  if (!FirstUse || &*std::next(MI.getIterator()) == FirstUse) {
    LLVM_DEBUG(dbgs() << "  No first use, or already just before it.\n");
    return false;
  }

  // Nothing crossed may conflict with the registers or memory MI touches.
  for (auto It = std::next(MI.getIterator()); It != FirstUse->getIterator();
       ++It) {
    MachineInstr &I = *It;
    if (I.isDebugInstr())
      continue;
    if (I.isCall() || I.hasUnmodeledSideEffects())
      return false;
    if (MI.mayLoad() && I.mayStore())
      return false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg())
        continue;
      for (const MachineOperand &IO : I.operands()) {
        if (!IO.isReg() || !IO.getReg())
          continue;
        Register A = MO.getReg(), B = IO.getReg();
        bool Overlap = A == B || (A.isPhysical() && B.isPhysical() &&
                                  TRI->regsOverlap(A, B));
        if (Overlap && (MO.isDef() || IO.isDef())) {
          LLVM_DEBUG(dbgs() << "  Conflict with: " << I);
          return false;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Sink " << MI << "  before " << *FirstUse);
  LLVM_DEBUG(dbgs() << "  Between: " << N << "\n");

  // Predict the effect on the vector register pressure and live interval
  // lengths. Sinking MI moves its DEF (and its LAST USEs) N instructions
  // later, so:
  //  - the N instructions it passes lose MI's net change (+DEF, -LAST USE),
  //  - MI's own point gains the net change of the N instructions passed,
  //    which are the DEFs and LAST USEs found among the events in the window,
  //  - the registers of MI: a first reference moves N instructions later (the
  //    interval gets shorter), a last reference moves later (longer).
  BlockUsage &BU = Usage[MBB];
  unsigned P = BU.Pos.lookup(&MI);
  unsigned Rows = BU.VecPrefix[P + N + 1] - BU.VecPrefix[P + 1];
  auto Weight = [&](Register R) {
    return (int)TRI->getRegClassWeight(MRI->getRegClass(R)).RegWeight;
  };

  // What every instruction MI passes gains in RP: MI's DEFs are not live yet
  // (-LMUL each) and the registers of its LAST USEs are still live (+LMUL each).
  int DeltaMI = 0;
  for (unsigned I = BU.NextEvent[P];
       I < BU.Events.size() && BU.Events[I].Pos == P; ++I)
    if (BU.Events[I].Pressure)
      DeltaMI += BU.Events[I].IsDef ? -BU.Events[I].W : BU.Events[I].W; // update passby RP

  // LIL is counted in rows (instructions touching a vector register) and MI is
  // one row. The registers of MI: a first reference moves Rows later (shorter
  // by Rows * LMUL), a last reference moves Rows later (longer).
  DenseMap<Register, int> LILChange;
  SmallSet<Register, 4> Seen; // a register can be an operand more than once
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual() ||
        !isVectorReg(MO.getReg(), *MRI) || !Seen.insert(MO.getReg()).second)
      continue;
    Register R = MO.getReg();
    auto Range = BU.Range.lookup(R);
    if (Range.first == P)
      LILChange[R] -= (int)Rows * Weight(R); // update sunk LIL
    if (Range.second == P)
      LILChange[R] += (int)Rows * Weight(R);
  }

  // One walk over the DEFs and LAST USEs in (P, P + N], starting at the first
  // event after MI and visiting no other instruction. Each one updates both:
  //  - RP: MI's point gains a DEF and loses a LAST USE it passes,
  //  - LIL: MI is one row that moves from before to after the window.
  //    A DEF whose register is used after the window now has MI between its
  //    first and last reference (+LMUL). A LAST USE whose register was first
  //    referenced before MI no longer has MI between them (-LMUL).
  // A register MI reads whose last use is in the window is now last used by
  // MI, so it stays live through the rest of the window: its LIL grows by the
  // rows after that last use and the RP of those instructions by its LMUL.
  SmallSet<Register, 4> MIUses;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isUse() && !MO.isUndef() && MO.getReg().isVirtual() &&
        isVectorReg(MO.getReg(), *MRI))
      MIUses.insert(MO.getReg());
  SmallVector<std::pair<unsigned, int>, 4> Extend; // (position, LMUL)
  int WindowNet = 0;
  unsigned NumEvents = 0;
  for (unsigned I = BU.NextEvent[P + 1];
       I < BU.Events.size() && BU.Events[I].Pos <= P + N; ++I) {
    const BlockUsage::RefEvent &E = BU.Events[I];
    ++NumEvents;
    if (E.Pressure)
      WindowNet += E.IsDef ? E.W : -E.W; // update sunk RP
    if (E.Reg.isVirtual()) {
      auto Range = BU.Range.lookup(E.Reg);
      if (E.IsDef && Range.second > P + N) // R's last use is after the window
        LILChange[E.Reg] += E.W; // update passby LIL
      if (!E.IsDef && MIUses.count(E.Reg)) {
        LILChange[E.Reg] += // update passby LIL, sunk is not last use but becomes last use after sink
            E.W * (int)(BU.VecPrefix[P + N + 1] - BU.VecPrefix[E.Pos + 1]);
        Extend.push_back({E.Pos, E.W});
      } else if (!E.IsDef && Range.first < P) { // R's first reference is before MI
        LILChange[E.Reg] -= E.W;
      }
    }
    LLVM_DEBUG(dbgs() << "    " << (E.IsDef ? "DEF" : "LAST USE") << " of "
                      << printReg(E.Reg, TRI) << " at +" << E.Pos - P
                      << ", LMUL " << E.W << "\n");
  }
  int LILDelta = 0;
  unsigned LILBefore = 0;
  for (auto &KV : BU.LIL)
    LILBefore += KV.second;
  for (auto &KV : LILChange)
    LILDelta += KV.second;
  LLVM_DEBUG(dbgs() << "  RP: passed instructions " << DeltaMI
                    << " each, MI point " << WindowNet << " from " << NumEvents
                    << " events\n");

  MBB->splice(FirstUse->getIterator(), MBB, MI.getIterator());
  if (LISValid)
    LIS->handleMove(MI);
  for (MachineOperand &MO : MI.all_uses())
    RegsToClearKillFlags.insert(MO.getReg());

  // Apply the prediction. The N passed instructions are now just before MI.
  unsigned OldRP = VRPressure.lookup(&MI);
  auto Passed = MI.getIterator();
  for (unsigned i = 0; i < N;) {
    --Passed;
    if (Passed->isDebugInstr())
      continue;
    unsigned Q = P + N - i; // position it had before the move
    VRPressure[&*Passed] += DeltaMI;
    for (auto &X : Extend)
      if (X.first <= Q)
        VRPressure[&*Passed] += X.second; // update passby RP, sunk is not last use but becomes last use after sink
    ++i;
  }
  VRPressure[&MI] += WindowNet;
  LLVM_DEBUG(dbgs() << "  RP of MI: " << OldRP << " -> " << VRPressure[&MI]
                    << "\n");
  DenseMap<Register, unsigned> PredictedLIL;
  for (auto &E : LILChange)
    PredictedLIL[E.first] = BU.LIL.lookup(E.first) + E.second;
  unsigned PredictedSLIL = LILBefore + LILDelta;

  computeBlockUsage(*MBB);

  // Verify against a fresh RegPressureTracker run and the recomputed lengths.
  LLVM_DEBUG({
    if (LISValid) {
      DenseMap<const MachineInstr *, unsigned> Fresh;
      computeBlockPressure(*MBB, Fresh);
      unsigned Bad = 0;
      for (MachineInstr &I : *MBB) {
        if (I.isDebugInstr())
          continue;
        if (Fresh.lookup(&I) != VRPressure.lookup(&I)) {
          dbgs() << "  RP MISMATCH at " << I << "    predicted "
                 << VRPressure.lookup(&I) << ", RPTracker " << Fresh.lookup(&I)
                 << "\n";
          ++Bad;
        }
      }
      if (!Bad)
        dbgs() << "  RP verified against RPTracker: OK\n";
      for (auto &KV : Fresh)
        VRPressure[KV.first] = KV.second;
    } else {
      dbgs() << "  RP verification skipped: LiveIntervals are stale\n";
    }
    for (auto &E : PredictedLIL) {
      unsigned Now = Usage[MBB].LIL.lookup(E.first);
      dbgs() << "  LIL of " << printReg(E.first, TRI) << ": "
             << (int)E.second - LILChange[E.first] << " -> predicted "
             << E.second << ", recomputed " << Now
             << (Now == E.second ? "  OK\n" : "  MISMATCH\n");
    }
    unsigned SLILNow = computeSLIL(*MBB, *MRI, *TRI);
    dbgs() << "  SLIL: " << LILBefore << " -> predicted " << PredictedSLIL
           << ", computeSLIL " << SLILNow
           << (SLILNow == PredictedSLIL ? "  OK\n" : "  MISMATCH\n");
  });
  return true;
}

// Sink the group (load, its mask COPY and passthru setup) down so that it ends
// up two instructions before the first use of the load result:
//   def A
//   def E
//   C = A * 2
bool ExpandPseudos::FindFirstUseToSinkToGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  FindFirstUseToSinkToGroup with "
                    << InstrsToSink.size() << " instructions.\n");
  if (InstrsToSink.empty())
    return false;
  MachineInstr &Primary = *InstrsToSink[0];
  MachineBasicBlock *MBB = Primary.getParent();
  for (MachineInstr *MI : InstrsToSink)
    if (MI->getParent() != MBB)
      return false;
  Register Reg = Primary.getOperand(0).getReg();

  unsigned Between = 0;
  MachineInstr *FirstUse = findFirstUseInBlock(Primary, Reg, *MRI, &Between);
  if (!FirstUse) {
    LLVM_DEBUG(dbgs() << "  First use not found.\n");
    return false;
  }

  MachineBasicBlock::iterator Target = FirstUse->getIterator();
  do {
    --Target;
  } while (Target->isDebugInstr());
  if (Between <= 2){
    LLVM_DEBUG(dbgs() << "  Already within 2 units of first use.\n");
    return false;
  } else {
    LLVM_DEBUG(dbgs()<<"  Between: "<<Between<<"\n");
  }

  unsigned Dist = 0;
  for (auto It = std::next(Primary.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    ++Dist;
    if (It == Target)
      break;
  }
  LLVM_DEBUG(dbgs() << "  Sink before: " << *Target);

  auto Overlaps = [&](Register A, Register B) {
    if (A == B)
      return true;
    return A.isPhysical() && B.isPhysical() && TRI->regsOverlap(A, B);
  };

  // Check that nothing crossed conflicts with the moved instructions.
  for (MachineInstr *M : InstrsToSink) {
    for (auto It = std::next(M->getIterator()); It != Target; ++It) {
      MachineInstr &I = *It;
      if (I.isDebugInstr())
        continue;
      if (I.isCall() || I.hasUnmodeledSideEffects())
        return false;
      if (M->mayLoad() && I.mayStore())
        return false;
      if (M->mayStore() && (I.mayLoad() || I.mayStore()))
        return false;
      for (const MachineOperand &MO : M->operands()) {
        if (!MO.isReg() || !MO.getReg())
          continue;
        for (const MachineOperand &IO : I.operands()) {
          if (!IO.isReg() || !IO.getReg() || !Overlaps(MO.getReg(), IO.getReg()))
            continue;
          // Def of M conflicts with any access; use of M conflicts with a def.
          if (MO.isDef() || IO.isDef()) {
            LLVM_DEBUG(dbgs() << "  Conflict with: " << I);
            return false;
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "  Sinking group of instructions\n");
  for (MachineInstr *MI : InstrsToSink) {
    LLVM_DEBUG(dbgs() << "  " << *MI);
  }
  MachineBasicBlock::iterator InsertPos = Target;
  for (auto It = InstrsToSink.begin(); It != InstrsToSink.end(); ++It) {
    MachineInstr *MI = *It;
    MachineBasicBlock::iterator CurrPos = MI->getIterator();
    MBB->splice(InsertPos, MBB, CurrPos);
    InsertPos = MI->getIterator();
    for (MachineOperand &MO : MI->all_uses())
      RegsToClearKillFlags.insert(MO.getReg());
  }
  LISValid = false;
  computeBlockUsage(*MBB);
  return true;
}

void ExpandPseudos::ProcessInSameBlock(MachineFunction &MF) {
  int64_t BlockSize = 128;
  for (const auto &BB : MF.getFunction()) {
    for (const auto &I : BB) {
      if (auto *VTy = dyn_cast<FixedVectorType>(I.getType())) {
        BlockSize = VTy->getNumElements();
      }
    }
  }
  LLVM_DEBUG(dbgs()<<"BlockSize: "<<BlockSize<<"\n");
  DenseSet<Register> SinkRegs;
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs()<<"ProcessInSameBlock.\n");
    AllSuccsCache AllSuccessors;

    // Sink LOAD
    // Walk the basic block bottom-up, as MachineSink.cpp does: sinking moves
    // an instruction later in the block, so processing from the end means a
    // sunk instruction is never revisited and never invalidates the
    // iterator for the instruction still to be processed.
    if (!MBB.empty()) {
      MachineBasicBlock::iterator It = std::prev(MBB.end());
      bool ProcessedBegin;
      do {
        MachineInstr &MI = *It;

        // Predecrement It (if it's not begin) so that it isn't invalidated by
        // sinking MI later in the block.
        ProcessedBegin = It == MBB.begin();
        if (!ProcessedBegin)
          --It;

        const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
        StringRef Name = TII->getName(MI.getOpcode());
        if (Name.contains("PseudoVMV_V_I")) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &MO = MI.getOperand(1);
          if (Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg()) {
            Register DstReg = Dst.getReg();
            if (!SinkRegs.count(DstReg)) {
              SinkRegs.insert(DstReg);
              LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
              if (FindFirstUseToSinkTo(MI, AllSuccessors)) {
                LLVM_DEBUG(dbgs()<<"  Sink success.\n");
              }
            }
          }
        }
        if (Name.contains("PseudoVMSLE_VI_M")) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &Src = MI.getOperand(1);
          if (Dst.isReg() && Src.isReg()) {
            Register DstReg = Dst.getReg();
            if (!SinkRegs.count(DstReg)) {
              SinkRegs.insert(DstReg);
              LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
              if (FindFirstUseToSinkTo(MI, AllSuccessors)) {
                LLVM_DEBUG(dbgs()<<"  Sink success.\n");
              }
            }
          }
        }
        if (Name.contains("PseudoVLE32_V_M") || isWholeRegLoadName(Name)) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &Src = MI.getOperand(1);
          if (Dst.isReg() && Src.isReg()) {
            Register DstReg = Dst.getReg();
            if (!SinkRegs.count(DstReg) && !RematRegs.count(DstReg) &&
                MI.memoperands_begin() != MI.memoperands_end()) {
              const MachineMemOperand *MMO = *MI.memoperands_begin();
              if (const Value *PtrVal = MMO->getValue()) {
                bool ShouldSink = false;
                if (ShouldSink) {
                  SinkRegs.insert(DstReg);
                  LLVM_DEBUG(dbgs()<<"Prepare to sink Group "<<MI);
                  SmallVector<MachineInstr*, 4> InstructionsToSink;
                  InstructionsToSink.push_back(&MI);
                  Register LDReg = MI.getOperand(0).getReg();
                  Register MaskReg = 0;
                  for (MachineOperand &MO : MI.all_uses()) {
                    if (MO.isReg() && !MO.getReg().isVirtual()) {
                      LLVM_DEBUG(dbgs() <<"  MaskReg: " <<MO<<"\n");
                      MaskReg = MO.getReg();
                      break;
                    }
                  }
                  if (MaskReg) {
                    MachineInstr *MaskDef = nullptr;
                    for (auto It = MI.getIterator(); It != MI.getParent()->begin(); --It) {
                      MachineInstr &Candidate = *std::prev(It);
                      //LLVM_DEBUG(dbgs() <<"  Reverse to find Mask: " <<Candidate);
                      for (MachineOperand &DefMO : Candidate.all_defs()) {
                        if (DefMO.isReg() && DefMO.getReg() == MaskReg) {
                          MaskDef = &Candidate;
                          break;
                        }
                      }
                      if (MaskDef)
                        break;
                    }
                    StringRef Name = TII->getName(MaskDef->getOpcode());
                    if (MaskDef && Name.contains("COPY")) {
                      LLVM_DEBUG(dbgs() <<"  Mask: " <<*MaskDef);
                      InstructionsToSink.push_back(MaskDef);
                    }
                  }
                  for (MachineInstr &UseMI : MRI->def_instructions(LDReg)) {
                    if (&UseMI == &MI) continue;
                    StringRef Name = TII->getName(UseMI.getOpcode());
                    if (Name.contains("PseudoVMV_V_I_M8")) {
                      LLVM_DEBUG(dbgs() <<"  VMV setup: " <<UseMI);
                      InstructionsToSink.push_back(&UseMI);
                      break;
                    }
                  }
                  if (FindFirstUseToSinkToGroup(InstructionsToSink, AllSuccessors)) {
                    LLVM_DEBUG(dbgs()<<"  Sink Group to first use success.\n");
                  }
                }
              }
            }
          }
        }
      } while (!ProcessedBegin);
    }
    SeenDbgUsers.clear();
    SeenDbgVars.clear();
    CachedRegisterPressure.clear();
  }
}

void ExpandPseudos::ProcessInSameAffine(MachineFunction &MF) {
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs() << "ProcessInSameAffine.\n");
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI = *I;
      LLVM_DEBUG(dbgs() << MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      if (MI.getOpcode() == RISCV::PseudoVID_V_M8) {
        MachineOperand &VIDDest = MI.getOperand(0);
        MachineOperand &VIDVL = MI.getOperand(2);
        MachineInstr *VOR_Orig = nullptr;
        Register VIDReg = VIDDest.getReg();
        Register VLReg = VIDVL.getReg();
        SmallVector<MachineInstr *, 8> Uses;
        for (MachineInstr &UseMI : MRI->use_instructions(VIDReg)) {
          if (&UseMI == &MI) continue;
          if (UseMI.getOpcode() == RISCV::PseudoVOR_VX_M8) {
            MachineOperand &SrcReg = UseMI.getOperand(2);
            if (SrcReg.isReg() && SrcReg.getReg() == VIDReg) {
              LLVM_DEBUG(dbgs() <<"  VOR_Orig: " <<UseMI);
              VOR_Orig = &UseMI;
            }
          } else if (UseMI.getOpcode() == RISCV::PseudoVADD_VX_M8) {
            MachineOperand &SrcReg = UseMI.getOperand(2);
            if (SrcReg.isReg() && SrcReg.getReg() == VIDReg) {
              LLVM_DEBUG(dbgs() <<"  Use VID result: " <<UseMI);
              Uses.push_back(&UseMI);
            }
          }
        }
        if (!VOR_Orig) {
          LLVM_DEBUG(dbgs() << "  No VOR_Orig found, skipping\n");
          continue;
        }
        // VOR_Orig would like to move up to right after MI (the VID), but
        // that is only safe if nothing between MI and VOR_Orig's current
        // position defines a register VOR_Orig reads (e.g. its scalar
        // operand, such as a base address computed after the VID but before
        // the OR): moving past such a def would use it before it is
        // defined. Instead of giving up in that case, move VOR_Orig up only
        // as far as right after the last such dependency, which is still
        // safe and still shortens the gap.
        MachineInstr *LastDep = nullptr;
        for (auto It = std::next(MI.getIterator());
             It != VOR_Orig->getIterator(); ++It) {
          for (const MachineOperand &Def : It->all_defs()) {
            if (!Def.isReg() || !Def.getReg())
              continue;
            for (const MachineOperand &Use : VOR_Orig->all_uses()) {
              if (Use.isReg() && Use.getReg() == Def.getReg()) {
                LastDep = &*It;
                break;
              }
            }
          }
        }
        MachineBasicBlock::iterator InsertPt =
            LastDep ? std::next(LastDep->getIterator())
                    : std::next(MI.getIterator());
        if (LastDep)
          LLVM_DEBUG(dbgs() << "  VOR_Orig depends on a value defined "
                                "between it and the VID, moving up only to "
                                "just after: " << *LastDep);
        if (InsertPt == VOR_Orig->getIterator()) {
          LLVM_DEBUG(dbgs() << "  VOR_Orig is already right after its last "
                                "dependency, nothing to move\n");
        } else {
          MBB.splice(InsertPt, &MBB, VOR_Orig->getIterator());
        }
        for (MachineInstr *UseMI : Uses) {
          MachineOperand &ImmReg = UseMI->getOperand(3);
          MachineInstr *ImmDef = MRI->getUniqueVRegDef(ImmReg.getReg());
          if (ImmDef && ImmDef->getOpcode() == RISCV::ADDI) {
            int64_t ImmValue = ImmDef->getOperand(2).getImm();
            LLVM_DEBUG(dbgs() <<"  VADD is Imm: " <<*UseMI);
            MachineInstr *VADD = UseMI;
            MachineInstr *VOR_ForVADD = nullptr;
            Register VADDDest = VADD->getOperand(0).getReg();
            for (MachineInstr &DestMI : MRI->use_instructions(VADDDest)) {
              if (DestMI.getOpcode() == RISCV::PseudoVOR_VX_M8) {
                MachineOperand &SrcReg = DestMI.getOperand(2);
                if (SrcReg.isReg() && SrcReg.getReg() == VADDDest) {
                  LLVM_DEBUG(dbgs() <<"  VOR_ForVADD: " <<DestMI);
                  VOR_ForVADD = &DestMI;
                  break;
                }
              }
            }
            if (!VOR_ForVADD) continue;
            Register BaseReg = VOR_Orig->getOperand(0).getReg();
            MachineInstr *NewVADD = nullptr;
            auto CreateNewVADD = [&](MachineInstr *OldVOR, MachineInstr *OldVADD) -> MachineInstr * {
              MachineInstrBuilder MIB = BuildMI(MBB, *OldVOR, OldVOR->getDebugLoc(),
                                                TII->get(RISCV::PseudoVADD_VX_M8));
              MIB.addReg(OldVOR->getOperand(0).getReg(), RegState::Define);
              MIB.addReg(OldVOR->getOperand(1).getReg(), RegState::Undef);
              MIB.addReg(BaseReg);
              MIB.addReg(OldVADD->getOperand(3).getReg());
              MIB.addReg(VLReg);
              MIB.addImm(5);
              MIB.addImm(1);
              MIB.copyImplicitOps(*OldVOR);
              return MIB;
            };
            NewVADD = CreateNewVADD(VOR_ForVADD, VADD);
            Register VORDest = VOR_ForVADD->getOperand(0).getReg();
            for (auto &DestMI : MRI->use_instructions(VORDest)) {
              if (DestMI.getOpcode() == RISCV::PseudoVMSLT_VX_M8) {
                MachineOperand &SrcReg = DestMI.getOperand(1);
                if (SrcReg.isReg() && SrcReg.getReg() == VORDest) {
                  LLVM_DEBUG(dbgs() <<"  VMSLT: " <<DestMI);
                  DestMI.getOperand(1).setReg(NewVADD->getOperand(0).getReg());
                  break;
                }
              }
            }
            VOR_ForVADD->eraseFromParent();
            VADD->eraseFromParent();
            LLVM_DEBUG(dbgs() << "  Replace VMSLT with NewVADD.\n");
          }
        }
        LLVM_DEBUG(dbgs() << "Transformed pattern from A to B.\n");
      }
    }
  }
}

// Two memory accesses are known not to alias if their underlying objects are
// distinct identified objects (e.g. different globals).
static bool isKnownNoAlias(const MachineInstr &A, const MachineInstr &B) {
  if (A.memoperands_empty() || B.memoperands_empty())
    return false;
  for (const MachineMemOperand *MA : A.memoperands())
    for (const MachineMemOperand *MB : B.memoperands()) {
      const Value *VA = MA->getValue(), *VB = MB->getValue();
      if (!VA || !VB)
        return false;
      const Value *OA = getUnderlyingObject(VA);
      const Value *OB = getUnderlyingObject(VB);
      if (OA == OB || !isIdentifiedObject(OA) || !isIdentifiedObject(OB))
        return false;
    }
  return true;
}

// Turn
//   %x = VL4RE32_V %p                           ; full load
//   ... (no store that may alias %p)
//   %x = PseudoVLE32_V_M4_MASK %x, %p, $v0, ... ; masked reload, identity on %x
//   ... uses of %x
// into a fresh full load of the same address defining a new vreg, so %x has a
// hole between its last use before the reload and the reload itself.
bool ExpandPseudos::ProcessRedundantReload(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      StringRef Name = TII->getName(MI.getOpcode());
      if (!Name.starts_with("PseudoVLE") || !Name.ends_with("_MASK"))
        continue;
      if (MI.getNumExplicitDefs() != 1 || MI.getNumOperands() < 3 ||
          MI.memoperands_empty())
        continue;
      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &Passthru = MI.getOperand(1);
      const MachineOperand &Base = MI.getOperand(2);
      if (!DstReg.isVirtual() || !Passthru.isReg() ||
          Passthru.getReg() != DstReg || !Base.isReg() ||
          !Base.getReg().isVirtual())
        continue;

      // %x must have exactly two defs: the earlier full load and this one.
      if (range_size(MRI->def_instructions(DstReg)) != 2)
        continue;
      MachineInstr *FullLoad = nullptr;
      for (MachineInstr &Def : MRI->def_instructions(DstReg))
        if (&Def != &MI)
          FullLoad = &Def;
      if (!FullLoad || FullLoad->getParent() != &MBB ||
          !isWholeRegLoadName(TII->getName(FullLoad->getOpcode())) ||
          FullLoad->getNumExplicitOperands() < 2 ||
          !FullLoad->getOperand(1).isReg() ||
          FullLoad->getOperand(1).getReg() != Base.getReg() ||
          FullLoad->memoperands_empty())
        continue;
      const MachineMemOperand *M0 = *FullLoad->memoperands_begin();
      const MachineMemOperand *M1 = *MI.memoperands_begin();
      if (M0->getValue() != M1->getValue() || M0->getOffset() != M1->getOffset())
        continue;

      // FullLoad must precede MI, and nothing in between may clobber memory.
      bool Ok = true, Seen = false;
      for (auto It = std::next(FullLoad->getIterator()); It != MI.getIterator();
           ++It) {
        if (It == MBB.end()) {
          Ok = false;
          break;
        }
        Seen = true;
        if (It->isDebugInstr())
          continue;
        if (It->isCall() || It->hasUnmodeledSideEffects() ||
            (It->mayStore() && !isKnownNoAlias(*It, *FullLoad))) {
          Ok = false;
          break;
        }
      }
      if (!Ok || !Seen)
        continue;

      // All uses of %x must be in this block; those after MI see the reload.
      SmallVector<MachineOperand *, 4> UsesAfter;
      bool AfterMI = false;
      for (MachineInstr &I : MBB) {
        if (&I == &MI) {
          AfterMI = true;
          continue;
        }
        if (!AfterMI)
          continue;
        for (MachineOperand &MO : I.operands())
          if (MO.isReg() && MO.getReg() == DstReg && MO.isUse())
            UsesAfter.push_back(&MO);
      }
      unsigned NumUses = range_size(MRI->use_operands(DstReg));
      // +1 for MI's own passthru use, which goes away.
      unsigned Before = NumUses - UsesAfter.size() - 1;
      for (MachineOperand &MO : MRI->use_operands(DstReg))
        if (MO.getParent()->getParent() != &MBB) {
          Ok = false;
          break;
        }
      if (!Ok)
        continue;
      (void)Before;

      LLVM_DEBUG(dbgs() << "Redundant reload: " << MI);
      Register NewReg = MRI->createVirtualRegister(MRI->getRegClass(DstReg));
      MachineInstr *NewLoad = MF.CloneMachineInstr(FullLoad);
      NewLoad->getOperand(0).setReg(NewReg);
      NewLoad->getOperand(0).setIsDead(false);
      NewLoad->getOperand(0).setIsDef();
      NewLoad->setDebugLoc(MI.getDebugLoc());
      MBB.insert(MI.getIterator(), NewLoad);
      RematRegs.insert(NewReg);
      for (MachineOperand *MO : UsesAfter) {
        MO->setReg(NewReg);
        MO->setIsKill(false);
      }
      RegsToClearKillFlags.insert(DstReg);
      RegsToClearKillFlags.insert(Base.getReg());
      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// Cheap instructions that may be recomputed instead of kept live: vector mask
// compares and immediate splats. They must be pure and define one register.
static bool isCheapALUOp(const MachineInstr &MI, StringRef Name) {
  if (!(Name.starts_with("PseudoVMS") || Name.starts_with("PseudoVMV_V_I")))
    return false;
  return !MI.mayLoadOrStore() && !MI.hasUnmodeledSideEffects() &&
         !MI.isCall() && MI.getNumExplicitDefs() == 1;
}

// Recompute a whole register load, or a cheap ALU op, right before a use that
// is far from the previous use, so the register is free in between:
//   %x = VL4RE32_V %p          (or %x = PseudoVMSLT_VV_M4 %a, %b)
//   use1 %x
//   use2 %x                    (loads need 2 uses here, ALU ops need 1)
//   ... >= RematGap instructions ...
//   other
//   use3 %x           -->   %y = <same instr> ; other ; use3 %y
// Every register the instruction reads must still be live at the new position,
// otherwise the recomputation would just extend another live range.
bool ExpandPseudos::ProcessRematLoads(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 8> Worklist;
    for (MachineInstr &MI : MBB) {
      StringRef Name = TII->getName(MI.getOpcode());
      if (isWholeRegLoadName(Name) || isCheapALUOp(MI, Name))
        Worklist.push_back(&MI);
    }

    while (!Worklist.empty()) {
      MachineInstr *Cand = Worklist.pop_back_val();
      bool IsLoad = Cand->mayLoad();
      if (Cand->getNumExplicitOperands() < 2 || !Cand->getOperand(0).isReg() ||
          (IsLoad && Cand->memoperands_empty()))
        continue;
      Register X = Cand->getOperand(0).getReg();
      if (!X.isVirtual() || !MRI->hasOneDef(X))
        continue;

      // Every register read must be virtual with a single def, so its value
      // is the same at the new position. Undef operands do not count.
      SmallVector<Register, 4> Inputs;
      bool InputsOk = true;
      for (const MachineOperand &MO : Cand->all_uses()) {
        if (!MO.isReg() || MO.isUndef() || MO.getReg() == X)
          continue;
        Register R = MO.getReg();
        if (!R.isVirtual() || !MRI->hasOneDef(R)) {
          InputsOk = false;
          break;
        }
        if (!is_contained(Inputs, R))
          Inputs.push_back(R);
      }
      if (!InputsOk || Inputs.empty())
        continue;

      // Collect the uses of X, requiring all of them to be in this block.
      SmallPtrSet<MachineInstr *, 8> UseSet;
      bool LocalOnly = true;
      for (MachineInstr &U : MRI->use_nodbg_instructions(X)) {
        if (U.getParent() != &MBB) {
          LocalOnly = false;
          break;
        }
        UseSet.insert(&U);
      }
      if (!LocalOnly || UseSet.size() < 2)
        continue;

      // Loads need the value to be reused before the gap, ALU ops are cheap.
      unsigned MinUses = IsLoad ? RematMinUses : 1;

      auto IsHazard = [&](const MachineInstr &I) {
        if (I.isCall() || I.hasUnmodeledSideEffects())
          return true;
        return IsLoad && I.mayStore() && !isKnownNoAlias(I, *Cand);
      };

      // Walk forward: look for the first use that is far from the previous.
      MachineInstr *PrevUse = nullptr;
      MachineInstr *Target = nullptr;
      unsigned Gap = 0, NumUses = 0;
      for (auto It = std::next(Cand->getIterator()); It != MBB.end(); ++It) {
        MachineInstr &I = *It;
        if (I.isDebugInstr())
          continue;
        if (UseSet.count(&I)) {
          if (NumUses >= MinUses && Gap >= RematGap) {
            Target = &I;
            break;
          }
          PrevUse = &I;
          ++NumUses;
          Gap = 0;
          continue;
        }
        if (!PrevUse)
          continue; // Instructions before the first use do not count as a gap.
        ++Gap;
      }
      if (!Target)
        continue;

      // Nothing between the original and the target may change memory.
      bool Hazard = false;
      for (auto It = std::next(Cand->getIterator());
           It != Target->getIterator(); ++It)
        if (!It->isDebugInstr() && IsHazard(*It)) {
          Hazard = true;
          break;
        }
      if (Hazard)
        continue;

      // Insert one non-debug instruction before the target so the clone has
      // some distance to the use: remat, another instr, use.
      MachineBasicBlock::iterator InsertPt(Target);
      do {
        --InsertPt;
      } while (InsertPt->isDebugInstr());

      // Only worth it if every input is live at the insertion point anyway.
      bool AllLive = true;
      for (Register R : Inputs) {
        bool Live = false;
        for (MachineOperand &MO : MRI->use_nodbg_operands(R)) {
          MachineInstr *U = MO.getParent();
          if (U == Cand)
            continue;
          if (U->getParent() != &MBB) {
            Live = true;
            break;
          }
          for (auto It = InsertPt; It != MBB.end(); ++It)
            if (&*It == U) {
              Live = true;
              break;
            }
          if (Live)
            break;
        }
        if (!Live) {
          AllLive = false;
          break;
        }
      }
      if (!AllLive) {
        LLVM_DEBUG(dbgs() << "Remat skipped, an input dies before target: "
                          << *Cand);
        continue;
      }

      LLVM_DEBUG(dbgs() << "Remat " << *Cand << "  before " << *InsertPt
                        << "  for use " << *Target);
      Register Y = MRI->createVirtualRegister(MRI->getRegClass(X));
      MachineInstr *NewMI = MF.CloneMachineInstr(Cand);
      for (MachineOperand &MO : NewMI->operands()) {
        if (!MO.isReg())
          continue;
        if (MO.getReg() == X) {
          // The def, and any tied undef use of it.
          MO.setReg(Y);
          if (MO.isDef())
            MO.setIsDead(false);
        } else if (MO.isUse()) {
          MO.setIsKill(false);
        }
      }
      NewMI->setDebugLoc(Target->getDebugLoc());
      MBB.insert(InsertPt, NewMI);
      RematRegs.insert(Y);

      // Rewrite Target and every later use of X.
      for (MachineInstr &I :
           make_range(MachineBasicBlock::iterator(Target), MBB.end()))
        for (MachineOperand &MO : I.operands())
          if (MO.isReg() && MO.isUse() && MO.getReg() == X) {
            MO.setReg(Y);
            MO.setIsKill(false);
          }
      RegsToClearKillFlags.insert(X);
      for (Register R : Inputs)
        RegsToClearKillFlags.insert(R);
      Changed = true;
      // The clone can be split again by a later distant use.
      Worklist.push_back(NewMI);
    }
  }
  return Changed;
}

// The last real (non-debug, non-self-def) use of R in its own block, or
// null. R must have a single definition. Used to anchor a reverse-remat
// insertion point right after the value it depends on is no longer needed
// for anything else, without searching from block-begin every time.
static MachineInstr *findLastRealUseInBlock(Register R,
                                             MachineRegisterInfo &MRI) {
  auto Defs = MRI.def_instructions(R);
  if (Defs.begin() == Defs.end())
    return nullptr;
  MachineInstr &Def = *Defs.begin();
  MachineInstr *Last = nullptr;
  for (auto It = std::next(Def.getIterator()); It != Def.getParent()->end();
       ++It) {
    if (It->isDebugInstr())
      continue;
    for (const MachineOperand &MO : It->operands())
      if (MO.isReg() && MO.isUse() && MO.getReg() == R) {
        Last = &*It;
        break;
      }
  }
  return Last;
}

// Reverse rematerialization (Bahi & Eisenbeis, "Register Reverse
// Rematerialization"): recompute a value from something derived from it,
// instead of keeping it alive, either as one hop:
//   %C = PseudoVFADD_VV_M8_E32 undef %C, %A, %B, frm, %vl, sew, policy, implicit $frm
//   ...
//   %I = <last use of %C>
//   %J = <uses %B again, much later>
// %B is live from its own definition all the way to %J. But since
// %C = %A + %B, %B can be reversibly recomputed from %C and %A as soon as
// %C is no longer needed for anything else, i.e. right after %C's last use:
//   %B2 = PseudoVFSUB_VV_M8_E32 undef %B2, %C, %A, frm, %vl, sew, policy, implicit $frm
// and %J's use of %B is rewritten to use %B2 instead. This shortens the live
// range of the original %B down to just [its def, %C's def], at the cost of
// one extra instruction and a new short-lived value %B2.
//
// Or, chained over several links of vector-scalar ops each invertible with
// the same scalar operand:
//   %B = PseudoVFADD_VFPR32_*(undef %B, %A, %k1, ...)   ; B = A + k1
//   %C = PseudoVFADD_VFPR32_*(undef %C, %B, %k2, ...)   ; C = B + k2
//   %D = PseudoVFMUL_VFPR32_*(undef %D, %C, %k3, ...)   ; D = C * k3
//   ...                                                  ; D's only forward
//   ...                                                  ; uses, no gap
//   %I = <uses %C again, much later>
//   %J = <uses %B again, much later>
//   %K = <uses %A again, much later>
// Each of %A, %B and %C is only needed once more, much later, so each can be
// reverse-rematerialized the same way as the single-hop case above: %C2 =
// %D / k3 replaces %C's late use, %B2 = %C2 - k2 replaces %B's late use,
// %A2 = %B2 - k1 replaces %A's late use. The key difference from just doing
// three independent single-hop replacements is that %B2 must be computed
// from %C2 (short-lived, still alive near %I), not from the original %C
// (which by then has died right after producing %D) -- so the chain must be
// walked from its deepest link (here, %D) backward, and each step's anchor
// is looked up through the replacements already made by the deeper steps.
bool ExpandPseudos::ProcessReverseRematChain(MachineFunction &MF) {
  // Forward opcode -> reverse opcode, for Y = OP(X, K) => X = REV(Y, K).
  // Note: integer PseudoVSLL_V*/PseudoVSRL_V* (shift) are deliberately not
  // listed here, unlike the pairs below. Shift is not a safe reverse op:
  // Y = X << k permanently discards X's top k bits, and Y = X >> k
  // permanently discards its bottom k bits, so recomputing X as Y reversed
  // by the opposite shift is only correct when those discarded bits happen
  // to be zero, which this opcode-level match cannot verify. Unlike the
  // float MUL/DIV pair below (exact in real arithmetic) or the ADD/SUB
  // pairs (exact under any fixed-width wraparound), a shift "reversal"
  // would silently compute wrong results for common inputs.
  static const std::pair<unsigned, unsigned> Reversible[] = {
      {RISCV::PseudoVFADD_VV_M4_E32, RISCV::PseudoVFSUB_VV_M4_E32},
      {RISCV::PseudoVFADD_VV_M8_E32, RISCV::PseudoVFSUB_VV_M8_E32},
      {RISCV::PseudoVFADD_VFPR32_M4_E32, RISCV::PseudoVFSUB_VFPR32_M4_E32},
      {RISCV::PseudoVFADD_VFPR32_M8_E32, RISCV::PseudoVFSUB_VFPR32_M8_E32},
      {RISCV::PseudoVFMUL_VFPR32_M4_E32, RISCV::PseudoVFDIV_VFPR32_M4_E32},
      {RISCV::PseudoVFMUL_VFPR32_M8_E32, RISCV::PseudoVFDIV_VFPR32_M8_E32},
      // Integer add: exact under 2's-complement wraparound for any k, so
      // safe to reverse the same way regardless of runtime values.
      {RISCV::PseudoVADD_VX_M4, RISCV::PseudoVSUB_VX_M4},
      {RISCV::PseudoVADD_VX_M8, RISCV::PseudoVSUB_VX_M8},
  };

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Original register -> its short-lived reverse-rematerialized
    // replacement, populated as deeper links in the chain are processed.
    DenseMap<Register, Register> ReplacedBy;
    // Walk the block from its end backward: the link closest to the end of
    // the chain (e.g. %D above) has no reverse-remat opportunity of its own
    // and is handled first, so ReplacedBy has what an earlier link (e.g.
    // %C's def) needs by the time it is processed.
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      unsigned RevOpcode = 0;
      for (auto &KV : Reversible)
        if (MI.getOpcode() == KV.first) {
          RevOpcode = KV.second;
          break;
        }
      // Minimum operand count for dst, undef passthru, vs2, rs1/vs1, and at
      // least vl/sew/policy: 7 for integer VX forms, 9 for the FP forms
      // (which also have frm before vl and an implicit $frm at the end).
      // The tail-copying loop below adapts to either shape automatically.
      if (!RevOpcode || MI.getNumOperands() < 7 || !MI.getOperand(0).isReg() ||
          !MI.getOperand(2).isReg() || !MI.getOperand(3).isReg())
        continue;

      Register Y = MI.getOperand(0).getReg();
      Register Vs2 = MI.getOperand(2).getReg();
      Register Vs1 = MI.getOperand(3).getReg();
      if (!Y.isVirtual() || !Vs2.isVirtual())
        continue;

      // A vector-vector op (e.g. C = A + B) is commutative in which operand
      // plays X (the one being reversed) vs. K (reused as-is in the reverse
      // op): either could be the one with a later second use. A
      // vector-scalar op's Vs1 is always a scalar FPR, never a value worth
      // reverse-rematerializing, so only Vs2 is tried there.
      SmallVector<std::pair<Register, Register>, 2> Candidates;
      Candidates.push_back({Vs2, Vs1});
      if (Vs1.isVirtual() && isVectorReg(Vs1, *MRI))
        Candidates.push_back({Vs1, Vs2});

      for (auto [X, K] : Candidates) {
        // X must have exactly two real uses: defining Y here, and one other
        // instruction. A tied self-use from a redefinition of X (e.g. a
        // masked load's mask-undisturbed passthru) does not count.
        unsigned NumRealUses = 0;
        MachineInstr *UseX = nullptr;
        for (MachineInstr &U : MRI->use_nodbg_instructions(X)) {
          bool IsOwnDef = false;
          for (const MachineOperand &Def : U.all_defs())
            if (Def.getReg() == X) {
              IsOwnDef = true;
              break;
            }
          if (IsOwnDef)
            continue;
          ++NumRealUses;
          if (&U != &MI)
            UseX = &U;
        }
        if (NumRealUses != 2 || !UseX)
          continue;

        // Anchor on Y's current representative: if a deeper link already
        // reverse-rematerialized Y, its short-lived replacement is the one
        // actually alive late in the block; Y itself may have died right
        // after this instruction.
        Register Anchor = ReplacedBy.lookup(Y);
        if (!Anchor)
          Anchor = Y;
        MachineInstr *LastUseOfAnchor = findLastRealUseInBlock(Anchor, *MRI);
        if (!LastUseOfAnchor)
          continue;
        // UseX must come strictly after the anchor's last use: the new
        // instruction is inserted right after LastUseOfAnchor, so if UseX
        // were that same instruction (e.g. I = C + F, reversing C anchored
        // on F, whose own only use is that same I), the insertion would
        // need to land both before and after UseX, which is impossible.
        bool AnchorBeforeUseX = false;
        for (auto It = std::next(LastUseOfAnchor->getIterator());
             It != MBB.end(); ++It)
          if (&*It == UseX) {
            AnchorBeforeUseX = true;
            break;
          }
        if (!AnchorBeforeUseX)
          continue;

        LLVM_DEBUG(dbgs() << "ReverseRematChain: recompute "
                          << printReg(X, TRI) << " from "
                          << printReg(Anchor, TRI) << " after "
                          << *LastUseOfAnchor << "  for use in " << *UseX);

        Register NewX = MRI->createVirtualRegister(MRI->getRegClass(X));
        MachineBasicBlock::iterator InsertPt =
            std::next(LastUseOfAnchor->getIterator());
        MachineInstrBuilder MIB =
            BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(RevOpcode));
        MIB.addReg(NewX, RegState::Define);
        MIB.addReg(NewX, RegState::Undef);
        MIB.addReg(Anchor);
        MIB.addReg(K);
        for (unsigned i = 4, e = MI.getNumOperands(); i != e; ++i)
          MIB.add(MI.getOperand(i));
        MIB->setFlags(MI.getFlags());

        for (MachineOperand &MO : UseX->operands())
          if (MO.isReg() && MO.isUse() && MO.getReg() == X)
            MO.setReg(NewX);

        RegsToClearKillFlags.insert(X);
        RegsToClearKillFlags.insert(Anchor);
        RegsToClearKillFlags.insert(K);
        ReplacedBy[X] = NewX;
        LISValid = false;
        Changed = true;
      }
    }
  }
  return Changed;
}

// A simple vector ALU op that can be a link in a rematerialization chain:
// %Y = PseudoV<MNEMONIC>_(VV|VX|VI|VFPR32)_M<n>[_E<sew>] undef %Y, %X, %K, ...
// i.e. exactly the operand shape ProcessReverseRematChain and
// ProcessForwardRematChain both key off (dst, undef passthru, vs2, then a
// second operand that is reused as-is, whether register or immediate).
// Matched by mnemonic prefix rather than a fixed opcode list, so it covers
// LMUL4/M8, integer and float, and VV/VX/VI/VFPR32 forms uniformly; masked
// forms are excluded since their extra mask/passthru operands don't fit
// this shape.
static bool isChainLinkInstr(const TargetInstrInfo *TII,
                              const MachineInstr &MI) {
  static const char *Mnemonics[] = {
      "PseudoVADD_",  "PseudoVSUB_",  "PseudoVRSUB_", "PseudoVMUL_",
      "PseudoVDIV_",  "PseudoVAND_",  "PseudoVOR_",   "PseudoVXOR_",
      "PseudoVSLL_",  "PseudoVSRL_",  "PseudoVSRA_",  "PseudoVFADD_",
      "PseudoVFSUB_", "PseudoVFRSUB_", "PseudoVFMUL_", "PseudoVFDIV_",
  };
  StringRef Name = TII->getName(MI.getOpcode());
  if (Name.contains("MASK"))
    return false;
  bool KnownMnemonic = false;
  for (const char *M : Mnemonics)
    if (Name.starts_with(M)) {
      KnownMnemonic = true;
      break;
    }
  return KnownMnemonic && MI.getNumOperands() >= 4 &&
         MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
         MI.getOperand(1).isUndef() && MI.getOperand(2).isReg();
}

// Figure 2(b) of the paper ("multiple instruction rematerialization"):
// instead of storing an intermediate value across a long gap (the problem
// ProcessReverseRematChain also solves, by reverse-computing it from
// something derived *later*), replay the *original* forward chain of
// instructions from a root value that is already live for the whole region
// anyway (e.g. a loaded input also needed again at the very end) to
// recompute the value fresh right where it is next needed:
//   %R = <root, e.g. a load, still needed much later at some other use>
//   %B = PseudoVADD_VX_M8 undef %B, %R, %k1, ...   ; B = R + k1
//   %C = PseudoVADD_VX_M8 undef %C, %B, %k2, ...   ; C = B + k2
//   ...
//   %I = <uses %C again, much later>
//   %J = <uses %B again, much later>
// becomes, right before each late use, a fresh replay of the chain from %R:
//   %B2 = PseudoVADD_VX_M8 undef %B2, %R, %k1, ...            ; before %J
//   %J = <rewritten to use %B2>
//   %Ca = PseudoVADD_VX_M8 undef %Ca, %R, %k1, ...            ; before %I
//   %C2 = PseudoVADD_VX_M8 undef %C2, %Ca, %k2, ...
//   %I = <rewritten to use %C2>
// Unlike reverse computation, this needs no algebraic inverse, so it is
// exact for *any* chain of ops -- including non-invertible ones like
// shifts or bitwise ops that ProcessReverseRematChain cannot touch -- at
// the cost of redoing however many steps lie between the root and the
// value, redundantly across different late uses if their chains overlap.
// The paper explicitly accepts that cost: "we don't consider this
// tradeoff and consider computation is free".
bool ExpandPseudos::ProcessForwardRematChain(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isDebugInstr() || !isChainLinkInstr(TII, MI))
        continue;

      // The candidate to recompute is normally the vs2 operand (Vs2), but
      // for a vector-vector op either operand could be the one with a
      // later second use (as in ProcessReverseRematChain); a vector-scalar
      // or vector-immediate op only ever has Vs2 as a candidate.
      Register Vs2 = MI.getOperand(2).getReg();
      SmallVector<Register, 2> Candidates;
      Candidates.push_back(Vs2);
      if (MI.getOperand(3).isReg()) {
        Register Vs1 = MI.getOperand(3).getReg();
        if (Vs1.isVirtual() && isVectorReg(Vs1, *MRI))
          Candidates.push_back(Vs1);
      }

      for (Register X : Candidates) {
        if (!X.isVirtual())
          continue;
        // X must have exactly two real uses: defining this link, and one
        // other, later instruction -- the same self-def exclusion as
        // ProcessReverseRematChain, for masked-redefinition passthrus.
        unsigned NumRealUses = 0;
        MachineInstr *UseX = nullptr;
        for (MachineInstr &U : MRI->use_nodbg_instructions(X)) {
          bool IsOwnDef = false;
          for (const MachineOperand &Def : U.all_defs())
            if (Def.getReg() == X) {
              IsOwnDef = true;
              break;
            }
          if (IsOwnDef)
            continue;
          ++NumRealUses;
          if (&U != &MI)
            UseX = &U;
        }
        if (NumRealUses != 2 || !UseX || UseX->getParent() != &MBB)
          continue;

        // Walk X's own ancestor chain of chain-link defs back to a root
        // that is not itself a chain-link result (e.g. a load). An empty
        // chain means X already *is* such a root: nothing cheaper to
        // replay it from, so there is nothing to do.
        SmallVector<MachineInstr *, 8> Chain; // built X-to-root, used root-to-X
        Register Cur = X;
        bool Ok = true;
        while (Chain.size() <= 8) {
          if (!Cur.isVirtual()) {
            Ok = false;
            break;
          }
          auto Defs = MRI->def_instructions(Cur);
          if (Defs.begin() == Defs.end()) {
            Ok = false;
            break;
          }
          MachineInstr &Def = *Defs.begin();
          if (!isChainLinkInstr(TII, Def) || Def.getParent() != &MBB)
            break; // Cur is the root.
          Chain.push_back(&Def);
          Cur = Def.getOperand(2).getReg();
        }
        if (!Ok || Chain.empty())
          continue;
        Register Root = Cur;
        std::reverse(Chain.begin(), Chain.end());

        // Only worth it if Root is already live at UseX anyway (another
        // use there or later, or in a different block): otherwise
        // replaying the chain would just extend Root's own live range
        // instead of shrinking anything, the same profitability check
        // ProcessRematLoads makes.
        bool RootLive = false;
        for (MachineOperand &MO : MRI->use_nodbg_operands(Root)) {
          MachineInstr *U = MO.getParent();
          if (U->getParent() != &MBB) {
            RootLive = true;
            break;
          }
          for (auto It = UseX->getIterator(); It != MBB.end(); ++It)
            if (&*It == U) {
              RootLive = true;
              break;
            }
          if (RootLive)
            break;
        }
        if (!RootLive)
          continue;

        LLVM_DEBUG(dbgs() << "ForwardRematChain: replay " << Chain.size()
                          << " step(s) from " << printReg(Root, TRI)
                          << " to recompute " << printReg(X, TRI)
                          << " for use in " << *UseX);

        Register Prev = Root;
        for (MachineInstr *Link : Chain) {
          bool IsLast = Link == Chain.back();
          Register Dst = MRI->createVirtualRegister(
              IsLast ? MRI->getRegClass(X)
                     : MRI->getRegClass(Link->getOperand(0).getReg()));
          MachineInstrBuilder MIB = BuildMI(MBB, UseX->getIterator(),
                                            Link->getDebugLoc(),
                                            TII->get(Link->getOpcode()));
          MIB.addReg(Dst, RegState::Define);
          MIB.addReg(Dst, RegState::Undef);
          MIB.addReg(Prev);
          MIB.add(Link->getOperand(3));
          for (unsigned i = 4, e = Link->getNumOperands(); i != e; ++i)
            MIB.add(Link->getOperand(i));
          MIB->setFlags(Link->getFlags());
          Prev = Dst;
        }

        for (MachineOperand &MO : UseX->operands())
          if (MO.isReg() && MO.isUse() && MO.getReg() == X)
            MO.setReg(Prev);

        RegsToClearKillFlags.insert(X);
        RegsToClearKillFlags.insert(Root);
        LISValid = false;
        Changed = true;
      }
    }
  }
  return Changed;
}

// Vector register pressure right after each instruction of MBB, tracked with
// RegPressureTracker as RISCVRegisterPressure does. Needs valid LiveIntervals.
void ExpandPseudos::computeBlockPressure(
    MachineBasicBlock &MBB, DenseMap<const MachineInstr *, unsigned> &Out) {
  if (VRSet == ~0u)
    return;
  IntervalPressure Pressure;
  RegPressureTracker RPTracker(Pressure);
  RPTracker.init(MBB.getParent(), &RegClassInfo, LIS, &MBB, MBB.begin(), false,
                 false);
  while (RPTracker.getPos() != MBB.end()) {
    const MachineInstr &MI = *RPTracker.getPos();
    RPTracker.advance();
    Out[&MI] = RPTracker.getRegSetPressureAtPos()[VRSet];
  }
}

// Positions, pressure events, register ranges and live interval lengths of
// the vector registers in MBB. The last use of a register does not count
// towards pressure, so a register is live after its first reference up to but
// not including its last one: +LMUL at the first, -LMUL at the last.
void ExpandPseudos::computeBlockUsage(MachineBasicBlock &MBB) {
  BlockUsage &BU = Usage[&MBB];
  for (auto &KV : BU.LIL)
    VRegLIL[KV.first] -= KV.second;
  BU = BlockUsage();

  DenseMap<Register, std::pair<unsigned, bool>> FirstIsUse; // first pos, is use
  DenseSet<Register> HasDef;
  DenseMap<Register, unsigned> OpenPhys; // physical vector reg -> last ref
  unsigned Pos = 0;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    BU.Pos[&MI] = Pos;
    BU.VecPrefix.push_back(BU.VecPrefix.empty() ? 0 : BU.VecPrefix.back());
    bool Touches = false;
    // Uses first, then defs, so a physical register redefined by the
    // instruction that also reads it is closed after the read.
    for (int Pass = 0; Pass < 2; ++Pass)
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || !isVectorReg(MO.getReg(), *MRI))
          continue;
        if (MO.isDef() != (Pass == 1))
          continue;
        // An undef use (the passthru tied to a def) reads nothing.
        if (MO.isUse() && MO.isUndef())
          continue;
        Touches = true;
        Register R = MO.getReg();
        if (R.isVirtual()) {
          auto It = BU.Range.try_emplace(R, Pos, Pos).first;
          It->second.second = Pos;
          FirstIsUse.try_emplace(R, Pos, MO.isUse());
          if (MO.isDef())
            HasDef.insert(R);
        } else if (MO.isDef()) {
          auto It = OpenPhys.find(R);
          if (It != OpenPhys.end())
            BU.Events.push_back({It->second, R, 1, false, true});
          BU.Events.push_back({Pos, R, 1, true, true});
          OpenPhys[R] = Pos;
        } else {
          auto It = OpenPhys.find(R);
          if (It != OpenPhys.end())
            It->second = Pos;
        }
      }
    if (Touches)
      ++BU.VecPrefix.back();
    ++Pos;
  }
  for (auto &KV : OpenPhys)
    BU.Events.push_back({KV.second, KV.first, 1, false, true});
  BU.VecPrefix.push_back(BU.VecPrefix.empty() ? 0 : BU.VecPrefix.back());

  for (auto &KV : BU.Range) {
    Register R = KV.first;
    unsigned W = TRI->getRegClassWeight(MRI->getRegClass(R)).RegWeight;
    // Live out of the block if referenced in another block, or carried around
    // a loop (first reference is a use and there is a def).
    bool LiveOut = FirstIsUse[R].second && HasDef.count(R);
    for (MachineInstr &U : MRI->reg_instructions(R))
      if (U.getParent() != &MBB) {
        LiveOut = true;
        break;
      }
    BU.Events.push_back({KV.second.first, R, (int)W, true, true});
    BU.Events.push_back({KV.second.second, R, (int)W, false, !LiveOut});
    BU.LIL[R] = (BU.VecPrefix[KV.second.second] - BU.VecPrefix[KV.second.first]) * W;
  }
  llvm::stable_sort(BU.Events, [](const BlockUsage::RefEvent &A,
                                  const BlockUsage::RefEvent &B) {
    return A.Pos < B.Pos;
  });
  BU.NextEvent.assign(Pos + 1, 0);
  for (unsigned I = 0, E = 0; I <= Pos; ++I) {
    while (E < BU.Events.size() && BU.Events[E].Pos < I)
      ++E;
    BU.NextEvent[I] = E;
  }
  for (auto &KV : BU.LIL)
    VRegLIL[KV.first] += KV.second;
}

// Compute the vector register pressure after each instruction and the live
// interval length of each vector vreg for the code as it is now.
void ExpandPseudos::computeVectorRegUsage(MachineFunction &MF,
                                          LiveIntervals &LI) {
  LIS = &LI;
  LISValid = true;
  VRPressure.clear();
  VRegLIL.clear();
  Usage.clear();

  VRSet = ~0u;
  for (unsigned i = 0, e = TRI->getNumRegPressureSets(); i != e; ++i)
    if (StringRef(TRI->getRegPressureSetName(i)) == "VR")
      VRSet = i;
  RegClassInfo.runOnMachineFunction(MF);

  for (MachineBasicBlock &MBB : MF) {
    computeBlockUsage(MBB);
    computeBlockPressure(MBB, VRPressure);
  }

  LLVM_DEBUG({
    unsigned MaxRP = 0, SLIL = 0;
    for (auto &KV : VRPressure)
      MaxRP = std::max(MaxRP, KV.second);
    for (auto &KV : VRegLIL)
      SLIL += KV.second;
    dbgs() << "Vector register usage: max VR pressure " << MaxRP << ", SLIL "
           << SLIL << "\n";
  });
}

// Peak vector-register pressure across the whole function and the sum of
// live-interval lengths (SLIL), computed purely from Usage/VRegLIL (a plain
// operand scan, kept up to date by every transform via computeBlockUsage).
// Unlike computeVectorRegUsage's own MaxRP (via RegPressureTracker), this
// does not need LiveIntervals, so it is safe to call after transforms that
// leave LISValid false, to see whether pressure and SLIL actually went down.
std::pair<unsigned, unsigned> ExpandPseudos::computeCurrentUsage() const {
  unsigned MaxPressure = 0;
  for (auto &BlockKV : Usage) {
    const BlockUsage &BU = BlockKV.second;
    unsigned Running = 0;
    for (unsigned I = 0, E = BU.Events.size(); I != E;) {
      unsigned Pos = BU.Events[I].Pos;
      unsigned J = I;
      while (J != E && BU.Events[J].Pos == Pos)
        ++J;
      // Apply this position's whole net delta (its DEFs and its own
      // operands' LAST USEs) atomically before checking the max: they
      // belong to the same instruction, whose destination can reuse a
      // dying source's register, so they are never simultaneously live.
      for (unsigned K = I; K != J; ++K)
        if (BU.Events[K].Pressure)
          Running += BU.Events[K].IsDef ? BU.Events[K].W : -BU.Events[K].W;
      MaxPressure = std::max(MaxPressure, Running);
      I = J;
    }
  }
  unsigned SLIL = 0;
  for (auto &KV : VRegLIL)
    SLIL += KV.second;
  return {MaxPressure, SLIL};
}

bool ExpandPseudos::runProcess(MachineFunction &MF) {

  bool EverMadeChange = false;

  if (Remat) {
    bool Changed = ProcessRedundantReload(MF);
    Changed |= ProcessRematLoads(MF);
    EverMadeChange |= Changed;
    if (Changed) {
      // Instructions were created without updating LiveIntervals.
      LISValid = false;
      for (MachineBasicBlock &MBB : MF)
        computeBlockUsage(MBB);
    }
  }
  if (Reverse) {
    bool Changed = ProcessReverseRematChain(MF);
    EverMadeChange |= Changed;
    if (Changed) {
      // Instructions were created without updating LiveIntervals.
      LISValid = false;
      for (MachineBasicBlock &MBB : MF)
        computeBlockUsage(MBB);
    }
  }
  if (Forward) {
    bool Changed = ProcessForwardRematChain(MF);
    EverMadeChange |= Changed;
    if (Changed) {
      // Instructions were created without updating LiveIntervals.
      LISValid = false;
      for (MachineBasicBlock &MBB : MF)
        computeBlockUsage(MBB);
    }
  }
  if (Sink)
    ProcessInSameBlock(MF);
  if (a) {
    ProcessInSameAffine(MF);
  }

  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();
  RematRegs.clear();

  return EverMadeChange;
}

bool ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "******** Expand Pseudos ********\n");
  if (!Sink && !a && !Remat && !Reverse && !Forward) {
    dbgs()<<"Custom Sink disabled.\n";
    return false;
  } else if (Sink) {
    std::string option = "";
    if (Sink) option += " Sink";
    if (Remat) option += " Remat";
    if (a) option += " Affine";
    if (Reverse) option += " Reverse";
    if (Forward) option += " Forward";
    dbgs()<<"Custom"<< option <<" enabled.\n";
  }

  STI = &MF.getSubtarget();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

  computeVectorRegUsage(MF, getAnalysis<LiveIntervalsWrapperPass>().getLIS());
  LLVM_DEBUG({
    auto [MaxRP, SLIL] = computeCurrentUsage();
    dbgs() << "Before transforms (LIS-independent): max VR pressure " << MaxRP
           << ", SLIL " << SLIL << "\n";
  });

  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      // Only expand pseudos.
      if (!MI.isPseudo())
        continue;

      // Give targets a chance to expand even standard pseudos.
      if (TII->expandPostRAPseudo(MI)) {
        LLVM_DEBUG(dbgs() << "expandPseudo: "<<MI);
        MadeChange = true;
        continue;
      }

      // Expand standard pseudos.
      switch (MI.getOpcode()) {
      //case TargetOpcode::SUBREG_TO_REG:
      //  MadeChange |= LowerSubregToReg(&MI);
      //  break;
      case TargetOpcode::COPY:
        MadeChange = LowerCopy(MBB, MI);
        if (MadeChange) {
          LLVM_DEBUG(dbgs()<<"lowerCopy success.\n");
        }
        break;
      case TargetOpcode::DBG_VALUE:
        continue;
      case TargetOpcode::INSERT_SUBREG:
      case TargetOpcode::EXTRACT_SUBREG:
        llvm_unreachable("Sub-register pseudos should have been eliminated.");
      }
    }
  }

  MadeChange |= runProcess(MF);

  LLVM_DEBUG({
    auto [MaxRP, SLIL] = computeCurrentUsage();
    dbgs() << "After transforms (LIS-independent): max VR pressure " << MaxRP
           << ", SLIL " << SLIL << "\n";
  });

  return MadeChange;
}
