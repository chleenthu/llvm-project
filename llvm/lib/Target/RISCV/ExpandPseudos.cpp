//===-- ExpandPseudos.cpp - Expand Pseudos ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpandPseudos.h"
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

ExpandPseudos::ExpandPseudos() : MachineFunctionPass(ID) {
  initializeExpandPseudosPass(*PassRegistry::getPassRegistry());
}

void ExpandPseudos::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  //AU.addRequired<MachineDominatorTreeWrapperPass>();
  //AU.addRequired<MachineCycleInfoWrapperPass>();
  //AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
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
            return true;
          }
        }
      }
    }
  }
  return MadeChange;
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
  MachineInstr *FirstUse = nullptr;
  for (MachineInstr &U : MRI->use_nodbg_instructions(Reg))
    if (U.getParent() != MBB)
      return false;
  for (auto It = std::next(MI.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    if (any_of(It->operands(), [&](const MachineOperand &MO) {
          return MO.isReg() && MO.isUse() && MO.getReg() == Reg;
        })) {
      FirstUse = &*It;
      break;
    }
  }
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
  MBB->splice(FirstUse->getIterator(), MBB, MI.getIterator());
  for (MachineOperand &MO : MI.all_uses())
    RegsToClearKillFlags.insert(MO.getReg());
  return true;
}

// Sink the group (load, its mask COPY and passthru setup) down so that it ends
// up two instructions before the first use of the load result:
//   def A
//   def E
//   C = A * 2
bool ExpandPseudos::FindFirstUseToSinkToGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  FindFirstUseToSinkToGroup.\n");
  if (InstrsToSink.empty())
    return false;
  MachineInstr &Primary = *InstrsToSink[0];
  MachineBasicBlock *MBB = Primary.getParent();
  for (MachineInstr *MI : InstrsToSink)
    if (MI->getParent() != MBB)
      return false;
  Register Reg = Primary.getOperand(0).getReg();

  // Find the first use and count the instructions between it and the load.
  MachineInstr *FirstUse = nullptr;
  unsigned Between = 0;
  for (auto It = std::next(Primary.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    if (any_of(It->operands(), [&](const MachineOperand &MO) {
          return MO.isReg() && MO.isUse() && MO.getReg() == Reg;
        })) {
      FirstUse = &*It;
      break;
    }
    ++Between;
  }
  if (!FirstUse) {
    LLVM_DEBUG(dbgs() << "  First use not found.\n");
    return false;
  }

  // Insert before the instruction right preceding the first use.
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

  // Already within 2 units of the insertion point: nothing to gain.
  unsigned Dist = 0;
  for (auto It = std::next(Primary.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    ++Dist;
    if (It == Target)
      break;
  }
  LLVM_DEBUG(dbgs() << "  Sink before: " << *Target);

  // Collect the group in block order so dependencies stay ordered.
  SmallPtrSet<MachineInstr *, 4> InGroup(InstrsToSink.begin(),
                                         InstrsToSink.end());
  SmallVector<MachineInstr *, 4> Ordered;
  for (MachineInstr &MI : *MBB) {
    if (InGroup.count(&MI))
      Ordered.push_back(&MI);
    if (&MI == &Primary)
      break;
  }
  if (Ordered.size() != InGroup.size())
    return false;

  auto Overlaps = [&](Register A, Register B) {
    if (A == B)
      return true;
    return A.isPhysical() && B.isPhysical() && TRI->regsOverlap(A, B);
  };

  // Check that nothing crossed conflicts with the moved instructions.
  for (MachineInstr *M : Ordered) {
    for (auto It = std::next(M->getIterator()); It != Target; ++It) {
      MachineInstr &I = *It;
      if (I.isDebugInstr() || InGroup.count(&I))
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
  for (MachineInstr *M : Ordered) {
    LLVM_DEBUG(dbgs() << "  " << *M);
    MBB->splice(Target, MBB, M->getIterator());
    for (MachineOperand &MO : M->all_uses())
      RegsToClearKillFlags.insert(MO.getReg());
  }
  return true;
}

MachineInstr *
ExpandPseudos::FindInSameSuccToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB,
                                 bool &BreakPHIEdge,
                                 AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs()<<"  FindInSameSuccToSinkTo.\n");
  assert(MBB && "Invalid MachineBasicBlock!");

  MachineInstr *SuccToSinkTo = nullptr;
  const MachineOperand &MO = MI.getOperand(0);
  Register Reg = MO.getReg();
  LLVM_DEBUG(dbgs()<<"  Dest: "<<MO<<"\n");
  //LLVM_DEBUG(dbgs()<<"UseMI:\n");
  bool AfterMI = false;
  for (MachineInstr &UseMI : *MBB) {
    if (&UseMI == &MI) {
      AfterMI = true;
      continue;
    }
    if (!AfterMI) {
      continue;
    }
    //LLVM_DEBUG(dbgs()<<"  "<<UseMI);
    if (UseMI.getNumOperands() > 0) {
      const MachineOperand &UseOp = UseMI.getOperand(0);
      if (UseOp.isReg() && UseOp.getReg() == Reg) {
        if (UseMI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
          LLVM_DEBUG(dbgs()<<"  UseMI: "<<UseMI);
          SuccToSinkTo = &UseMI;
          break;
        }
      }
    }
    if (UseMI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32) {
      for (unsigned i = 0; i < UseMI.getNumOperands(); i++) {
        const MachineOperand &UseOp = UseMI.getOperand(i);
        if (UseOp.isReg() && UseOp.getReg() == Reg) {
          LLVM_DEBUG(dbgs()<<"  UseMI: "<<UseMI);
          LLVM_DEBUG(dbgs() << "  "<<UseOp<<" is used as operand " << i << "\n");
          SuccToSinkTo = &UseMI;
          return SuccToSinkTo;
        }
      }
    }
    if (UseMI.getOpcode() == RISCV::PseudoVFMAX_VV_M8_E32) {
      for (unsigned i = 0; i < UseMI.getNumOperands(); i++) {
        const MachineOperand &UseOp = UseMI.getOperand(i);
        if (UseOp.isReg() && UseOp.getReg() == Reg) {
          LLVM_DEBUG(dbgs()<<"  UseMI: "<<UseMI);
          LLVM_DEBUG(dbgs() << "  "<<UseOp<<" is used as operand " << i << "\n");
          SuccToSinkTo = &UseMI;
          return SuccToSinkTo;
        }
      }
    }
  }
  return SuccToSinkTo;
}

bool ExpandPseudos::SinkInSameInstruction(MachineInstr &MI, bool &SawStore,
                                     AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs()<<"  SinkInSameInstruction.\n");
  if (!TII->shouldSink(MI)) {
    LLVM_DEBUG(dbgs()<<"should not sink.\n");
    return false;
  }
  if (!MI.isSafeToMove(SawStore)) {
    LLVM_DEBUG(dbgs()<<"not safe to move.\n");
    return false;
  }
  if (MI.isConvergent()) {
    LLVM_DEBUG(dbgs()<<"is convergent.\n");
    return false;
  }
  bool BreakPHIEdge = false;
  MachineBasicBlock *ParentBlock = MI.getParent();
  MachineBasicBlock *SuccToSinkTo = MI.getParent();
  MachineInstr *TargetUser =
      FindInSameSuccToSinkTo(MI, ParentBlock, BreakPHIEdge, AllSuccessors);
  if (!TargetUser)
    return false;
  for (const MachineOperand &MO : MI.all_defs()) {
    Register Reg = MO.getReg();
    if (Reg == 0 || !Reg.isPhysical())
      continue;
    if (SuccToSinkTo->isLiveIn(Reg))
      return false;
  }
  LLVM_DEBUG(dbgs() << "  Sink instr " << MI);
  MachineBasicBlock::iterator InsertPos = TargetUser->getIterator();
  MachineBasicBlock::iterator CurrPos = MI.getIterator();
  SmallVector<MIRegs, 4> DbgUsersToSink;
  for (auto &MO : MI.all_defs()) {
    if (!MO.getReg().isVirtual())
      continue;
    auto It = SeenDbgUsers.find(MO.getReg());
    if (It == SeenDbgUsers.end())
      continue;
    auto &Users = It->second;
    for (auto &User : Users) {
      MachineInstr *DbgMI = User.getPointer();
      if (User.getInt()) {
        if (!attemptDebugCopyProp(MI, *DbgMI, MO.getReg()))
          DbgMI->setDebugValueUndef();
      } else {
        DbgUsersToSink.push_back(
            {DbgMI, SmallVector<Register, 2>(1, MO.getReg())});
      }
    }
  }
  //performSink(MI, *SuccToSinkTo, InsertPos, DbgUsersToSink);
  SuccToSinkTo->splice(InsertPos, SuccToSinkTo, CurrPos);
  for (MachineOperand &MO : MI.all_uses())
    RegsToClearKillFlags.insert(MO.getReg());
  return true;
}

bool ExpandPseudos::SinkInSameInstructionGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, bool &SawStore,
    AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  SinkInSameInstructionGroup with "
                    << InstrsToSink.size() << " instructions.\n");
  if (InstrsToSink.empty())
    return false;
  MachineInstr &PrimaryMI = *InstrsToSink[0];
  for (MachineInstr *MI : InstrsToSink) {
    if (!TII->shouldSink(*MI)) {
      LLVM_DEBUG(dbgs() << "should not sink: " << *MI);
      return false;
    }
    if (!MI->isSafeToMove(SawStore)) {
      LLVM_DEBUG(dbgs() << "not safe to move: " << *MI);
      return false;
    }
    if (MI->isConvergent()) {
      LLVM_DEBUG(dbgs() << "is convergent: " << *MI);
      return false;
    }
  }
  bool BreakPHIEdge = false;
  MachineBasicBlock *ParentBlock = PrimaryMI.getParent();
  MachineBasicBlock *SuccToSinkTo = PrimaryMI.getParent();
  MachineInstr *TargetUser =
      FindInSameSuccToSinkTo(PrimaryMI, ParentBlock, BreakPHIEdge, AllSuccessors);
  if (!TargetUser)
    return false;
  for (MachineInstr *MI : InstrsToSink) {
    for (const MachineOperand &MO : MI->all_defs()) {
      Register Reg = MO.getReg();
      if (Reg == 0 || !Reg.isPhysical())
        continue;
      if (SuccToSinkTo->isLiveIn(Reg))
        return false;
    }
  }
  LLVM_DEBUG(dbgs() << "  Sinking group of instructions\n");
  for (MachineInstr *MI : InstrsToSink) {
    LLVM_DEBUG(dbgs() << "  " << *MI);
  }
  MachineBasicBlock::iterator InsertPos = TargetUser->getIterator();
  for (auto It = InstrsToSink.begin(); It != InstrsToSink.end(); ++It) {
    MachineInstr *MI = *It;
    MachineBasicBlock::iterator CurrPos = MI->getIterator();
    SuccToSinkTo->splice(InsertPos, SuccToSinkTo, CurrPos);
    InsertPos = MI->getIterator();
  }
  SmallVector<MIRegs, 4> DbgUsersToSink;
  for (MachineInstr *MI : InstrsToSink) {
    for (auto &MO : MI->all_defs()) {
      if (!MO.getReg().isVirtual())
        continue;
      auto It = SeenDbgUsers.find(MO.getReg());
      if (It == SeenDbgUsers.end())
        continue;
      auto &Users = It->second;
      for (auto &User : Users) {
        MachineInstr *DbgMI = User.getPointer();
        if (User.getInt()) {
          if (!attemptDebugCopyProp(*MI, *DbgMI, MO.getReg()))
            DbgMI->setDebugValueUndef();
        } else {
          DbgUsersToSink.push_back(
              {DbgMI, SmallVector<Register, 2>(1, MO.getReg())});
        }
      }
    }
    for (MachineOperand &MO : MI->all_uses())
      RegsToClearKillFlags.insert(MO.getReg());
  }
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
  int SinkVMVTime = 0;
  int LoadTime256 = 0;
  int LoadTime384 = 0;
  DenseSet<Register> SinkRegs;
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs()<<"ProcessInSameBlock.\n");
    AllSuccsCache AllSuccessors;
    // Sink VMV and MASK. Iterate backward so that SawStore only covers the
    // stores after the current instruction, which are the ones a sunk
    // instruction may cross. Sinking moves instructions later, to positions
    // that were already visited, so walk a snapshot of the block.
    SmallVector<MachineInstr *, 64> Snapshot;
    for (MachineInstr &I : MBB)
      Snapshot.push_back(&I);
    bool SawStoreBwd = false;
    for (MachineInstr *MIPtr : llvm::reverse(Snapshot)) {
      MachineInstr &MI = *MIPtr;
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      StringRef Name = TII->getName(MI.getOpcode());
      if (Name.contains("PseudoVMV_V_I")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &MO = MI.getOperand(1);
        ++SinkVMVTime;
        if (SinkVMVTime <= 2 && Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStoreBwd, AllSuccessors))
            LLVM_DEBUG(dbgs()<<"  Sink success.\n");
        }
      }
      if (Name.contains("PseudoVMSLE_VI_M")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStoreBwd, AllSuccessors))
            LLVM_DEBUG(dbgs()<<"  Sink success.\n");
        }
      }
      // Track stores seen so far (i.e. located after later-visited
      // instructions) for the isSafeToMove query.
      if (MI.mayStore())
        SawStoreBwd = true;
    }

    // Sink LOAD
    // Dont modify here. For easy stdout review I iterate forward here.
    //LLVM_DEBUG(dbgs()<<"==================== Sink LOAD ====================\n");
    bool ProcessedBegin, SawStore = false;
    for (auto It = MBB.begin(); It != MBB.end(); ){
      MachineInstr &MI = *It;
      bool Sunk = false;
      auto NextIt = std::next(It);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      StringRef Name = TII->getName(MI.getOpcode());
      if (Name.contains("PseudoVMV_V_I")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &MO = MI.getOperand(1);
        ++SinkVMVTime;
        if (SinkVMVTime <= 2 && Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"  Sink success.\n");
            Sunk = true;
          }
        }
      }
      if (Name.contains("PseudoVMSLE_VI_M")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (FindFirstUseToSinkTo(MI, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"  Sink success.\n");
            Sunk = true;
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
              /*int64_t Offset = MMO->getOffset();
              bool ShouldSink = false;
              if (Offset >= (BlockSize / 2) * 4) {
                LLVM_DEBUG(dbgs() << "  Matched 3rd VLE32 load: offset=" << Offset << "\n");
                ++LoadTime256;
                if (1 <= LoadTime256) {
                  ShouldSink = true;
                }
              }
              else if (Offset == (BlockSize / 4 * 3) * 4) {
                LLVM_DEBUG(dbgs() << "  Matched 4th VLE32 load: offset=" << Offset << "\n");
                ++LoadTime384;
                if (1 <= LoadTime384) {
                  ShouldSink = true;
                }
              }*/
              bool ShouldSink = true;
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
                if (SinkInSameInstructionGroup(InstructionsToSink, SawStore, AllSuccessors)) {
                  LLVM_DEBUG(dbgs()<<"  Sink Group success.\n");
                  Sunk = true;
                } else if (FindFirstUseToSinkToGroup(InstructionsToSink, AllSuccessors)) {
                  LLVM_DEBUG(dbgs()<<"  Sink Group to first use success.\n");
                  Sunk = true;
                }
              }
            }
          }
        }
      }
      if (Sunk) {
        It = NextIt;
      } else {
        ++It;
      }
    }// while (!ProcessedBegin);
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
        MBB.splice(std::next(MI.getIterator()), &MBB, VOR_Orig->getIterator());
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

bool ExpandPseudos::runProcess(MachineFunction &MF) {

  bool EverMadeChange = false;

  if (Remat) {
    EverMadeChange |= ProcessRedundantReload(MF);
    EverMadeChange |= ProcessRematLoads(MF);
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
  if (!Sink && !a && !Remat) {
    dbgs()<<"RISCV Sink disabled.\n";
    return false;
  } else if (Sink) {
    dbgs()<<"RISCV Sink enabled.\n";
  } else if (a) {
    dbgs()<<"RISCV Affine enabled.\n";
  }

  STI = &MF.getSubtarget();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

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

  return MadeChange;
}
