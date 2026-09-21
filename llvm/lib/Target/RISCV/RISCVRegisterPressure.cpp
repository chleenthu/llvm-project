#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

using namespace llvm;

#define DEBUG_TYPE "registerpressure"
#define RISCV_INSERT_VSETVLI_NAME "Custom RISCV-V Register Pressure pass"

namespace {
struct RegisterPressureHotSpot {
  MachineBasicBlock::const_iterator BeginPos;
  MachineBasicBlock::const_iterator EndPos;
  bool BeginClosed = false;
  bool EndClosed = false;
  bool BeginIsClosed() { return BeginClosed; }
  bool EndIsClosed() { return EndClosed; }
  bool IsClosed() { return BeginIsClosed() && EndIsClosed(); }
  void Init() {BeginClosed = EndClosed = false;}
};

static cl::opt<bool>
    EnableRegisterPressureOpt("custom-pressure",
        cl::init(true), cl::Hidden,
        cl::desc("Measure vector register pressure of loop blocks and sink "
                 "loads out of pressure hot spots"));

// Only blocks with more than this many instructions are measured.
static constexpr unsigned MinBlockInstrs = 20;

class RISCVRegisterPressure : public MachineFunctionPass {
public:
  static char ID;

  RISCVRegisterPressure() : MachineFunctionPass(ID) {
    initializeRISCVRegisterPressurePass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }
private:
  const MachineLoopInfo *MLI;
  LiveIntervals *LIS;

  bool enableRegisterPressureOpt() const;
  // get Loop MDNode
  MDNode *getLoopID(MachineLoop *ML) const;
  // get loop bool attribute
  std::optional<bool> getOptionalBoolLoopAttribute(MDNode *LoopID, StringRef Name);
  bool isVLoad(const MachineInstr &MI);
  bool hasHotSpotInLoop(MachineFunction &MF, MachineLoop *ML, RegisterPressureHotSpot &P);
  bool trySinkLoad(MachineFunction &MF, RegisterPressureHotSpot &P, std::vector<Register> &Regs, Register TargetReg);
};

}
char RISCVRegisterPressure::ID=0;
char &llvm::RISCVRegisterPressureID = RISCVRegisterPressure::ID;

INITIALIZE_PASS(RISCVRegisterPressure, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

bool RISCVRegisterPressure::enableRegisterPressureOpt() const {
  return EnableRegisterPressureOpt;
}

MDNode *RISCVRegisterPressure::getLoopID(MachineLoop *ML) const {
  MDNode *LoopID = nullptr;
  if (const auto *MBB = ML->findLoopControlBlock()) {
    // If there is a single latch block, then the metadata
    // node is attached to its terminating instruction.
    const auto *BB = MBB->getBasicBlock();
    if (!BB)
      return nullptr;
    if (const auto *TI = BB->getTerminator())
      LoopID = TI->getMetadata(LLVMContext::MD_loop);
  } else if (const auto *MBB = ML->getHeader()) {
    // There seem to be multiple latch blocks, so we have to
    // visit all predecessors of the loop header and check
    // their terminating instructions for the metadata.
    if (const auto *Header = MBB->getBasicBlock()) {
      // Walk over all blocks in the loop.
      for (const auto *MBB : ML->blocks()) {
        const auto *BB = MBB->getBasicBlock();
        if (!BB)
          return nullptr;
        const auto *TI = BB->getTerminator();
        if (!TI)
          return nullptr;
        MDNode *MD = nullptr;
        // Check if this terminating instruction jumps to the loop header.
        for (const auto *Succ : successors(TI)) {
          if (Succ == Header) {
            // This is a jump to the header - gather the metadata from it.
            MD = TI->getMetadata(LLVMContext::MD_loop);
            break;
          }
        }
        if (!MD)
          return nullptr;
        if (!LoopID)
          LoopID = MD;
        else if (MD != LoopID)
          return nullptr;
      }
    }
  }
  if (LoopID &&
      (LoopID->getNumOperands() == 0 || LoopID->getOperand(0) != LoopID))
    LoopID = nullptr;
  return LoopID;
}
std::optional<bool> RISCVRegisterPressure::getOptionalBoolLoopAttribute(MDNode *LoopID,
                                                       StringRef Name) {
  MDNode *MD = findOptionMDForLoopID(LoopID, Name);
  if (!MD)
    return std::nullopt;
  switch (MD->getNumOperands()) {
  case 1:
    // When the value is absent it is interpreted as 'attribute set'.
    return true;
  case 2:
    if (ConstantInt *IntMD =
            mdconst::extract_or_null<ConstantInt>(MD->getOperand(1).get()))
      return IntMD->getZExtValue();
    return true;
  }
  llvm_unreachable("unexpected number of options");
}
bool RISCVRegisterPressure::isVLoad(const MachineInstr &MI) {
  switch(MI.getOpcode()) {
    default:
      return false;
    case RISCV::VL1RE8_V:
    case RISCV::VL1RE16_V:
    case RISCV::VL1RE32_V:
    case RISCV::VL1RE64_V:
    case RISCV::VL2RE8_V:
    case RISCV::VL2RE16_V:
    case RISCV::VL2RE32_V:
    case RISCV::VL2RE64_V:
    case RISCV::VL4RE8_V:
    case RISCV::VL4RE16_V:
    case RISCV::VL4RE32_V:
    case RISCV::VL4RE64_V:
    case RISCV::VL8RE8_V:
    case RISCV::VL8RE16_V:
    case RISCV::VL8RE32_V:
    case RISCV::VL8RE64_V:
      return true;
  }
}
// Print "VR=<n> (GPR=<m>)", where GPR is the GPRAll pressure set. VMV0 is a subset of VR and is left out.
static void printVRAndGPRPressure(ArrayRef<unsigned> Pressure,
                                  const TargetRegisterInfo *TRI, bool showExplicit=false) {
  unsigned VR = 0, GPR = 0;
  for (unsigned i = 0, e = Pressure.size(); i != e; ++i) {
    StringRef Name = TRI->getRegPressureSetName(i);
    if (Name == "VR")
      VR = Pressure[i];
    else if (Name == "GPRAll")
      GPR = Pressure[i];
  }
  if (!showExplicit)
    LLVM_DEBUG(dbgs() << "VR=" << VR << "\n");//" (GPR=" << GPR << ")\n");
  else
    dbgs() << "VR=" << VR << "\n";//" (GPR=" << GPR << ")\n";
}

static std::pair<unsigned, unsigned>
getVRAndGPRPressure(ArrayRef<unsigned> Pressure, const TargetRegisterInfo *TRI) {
  unsigned VR = 0, GPR = 0;
  for (unsigned i = 0, e = Pressure.size(); i != e; ++i) {
    StringRef Name = TRI->getRegPressureSetName(i);
    if (Name == "VR")
      VR = Pressure[i];
    else if (Name == "GPRAll")
      GPR = Pressure[i];
  }
  return {VR, GPR};
}

// Sum of live interval lengths (SLIL) of the vector virtual registers used in
// the block. The last use of a register does not count towards register
// pressure, so the length of a register is (Last - First) * LMUL, where First
// and Last are the first and last instruction referencing it, counted in
// instructions that touch a vector register. For
//   def A; def B; C = A * 2
// the length of A is 2 * LMUL.
static unsigned computeSLIL(const MachineBasicBlock &MBB,
                            const MachineRegisterInfo &MRI,
                            const TargetRegisterInfo &TRI) {
  auto IsVector = [&](Register R) {
    if (R.isVirtual())
      return RISCVRI::isVRegClass(MRI.getRegClass(R)->TSFlags);
    return R.isPhysical() && RISCV::VRRegClass.contains(R);
  };
  DenseMap<Register, std::pair<unsigned, unsigned>> Range;
  unsigned Row = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    bool Touches = false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg() || !IsVector(MO.getReg()))
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
    unsigned LMUL =
        TRI.getRegClassWeight(MRI.getRegClass(KV.first)).RegWeight;
    Sum += (KV.second.second - KV.second.first) * LMUL;
  }
  return Sum;
}

bool RISCVRegisterPressure::hasHotSpotInLoop(MachineFunction &MF, MachineLoop *ML, RegisterPressureHotSpot &P) {
  P.Init();
  for (const MachineBasicBlock* B: ML->getBlocks()) {
    // Small blocks are not worth measuring.
    if (llvm::count_if(*B, [](const MachineInstr &MI) {
          return !MI.isDebugInstr();
        }) <= MinBlockInstrs)
      continue;
    //B->dump();
    // RegionPressure or IntervalPressure (need LIS)
    //RegionPressure Pressure;
    IntervalPressure Pressure;
    RegPressureTracker RPTracker(Pressure);
    RegisterClassInfo RegClassInfo;
    RegClassInfo.runOnMachineFunction(MF);
    RPTracker.init(&MF, &RegClassInfo, LIS, B, B->begin(),false, false);
    while (RPTracker.getPos() != B->end()) {
      auto MBI = RPTracker.getPos();
      auto MII = MBI.getInstrIterator();

      LLVM_DEBUG(dbgs() << "====================================================\n");
      LLVM_DEBUG(dbgs() << *MII);
      //if (!MII.isEnd())
      //if (MBI != B->begin()) {
      //  auto PMBI = prev_nodbg(MBI, B->begin());
      //  auto PMII = PMBI.getInstrIterator();
      //  LLVM_DEBUG(dbgs() << "RP: prev MI " << *PMII << "\n");
      //}
      //LLVM_DEBUG(dbgs() << "RP: cur MI " << *MII << "\n");
      auto PSet = RPTracker.getRegSetPressureAtPos();
      if (!P.BeginIsClosed() && PSet[14] > 32) {
        P.BeginPos = prev_nodbg(MBI, B->begin());;
        P.BeginClosed = true;
        //LLVM_DEBUG(dbgs() << "RP: P.BeginClosed set " << *P.BeginPos.getInstrIterator() << "\n");
      }
      if (P.BeginIsClosed() && !P.EndIsClosed() && PSet[14] <= 32) {
        P.EndPos = MBI;
        P.EndClosed = true;
        //LLVM_DEBUG(dbgs() << "RP: P.EndClosed set " << *P.EndPos.getInstrIterator() << "\n");

        // find a hot spot, so return

        return true;
      }


      RPTracker.advance();
      // Pressure after the instruction printed above.
      LLVM_DEBUG(dbgs() << "Curr Pressure: ");
      //dumpRegSetPressure(RPTracker.getRegSetPressureAtPos(),
      //                   MF.getSubtarget().getRegisterInfo());
      printVRAndGPRPressure(RPTracker.getRegSetPressureAtPos(),
                            MF.getSubtarget().getRegisterInfo());
    }

    RPTracker.closeBottom();
    LLVM_DEBUG(dbgs() << "====================================================\n");
    LLVM_DEBUG(dbgs() << "end of block\n");
    // RPTracker.dump() also prints Live In / Live Out, so print only Max.
    //RPTracker.dump();
    LLVM_DEBUG(dbgs() << "PRP: ");
    //dumpRegSetPressure(Pressure.MaxSetPressure,
    //                   MF.getSubtarget().getRegisterInfo());
    printVRAndGPRPressure(Pressure.MaxSetPressure,
                          MF.getSubtarget().getRegisterInfo());
    LLVM_DEBUG(dbgs() << "SLIL: " << computeSLIL(*B, MF.getRegInfo(), *MF.getSubtarget().getRegisterInfo()) << "\n");
    //LLVM_DEBUG(dbgs() << "isTopClosed:" << RPTracker.isTopClosed() << "\n");
    //LLVM_DEBUG(dbgs() << "isBottomClosed:" << RPTracker.isBottomClosed() << "\n");
    //RPTracker.closeRegion();
    //std::vector<unsigned> Pr, MPr;
    //RPTracker.getUpwardPressure(&(*(B->begin())), Pr, MPr);
    //dumpRegSetPressure(RPTracker.getRegSetPressureAtPos(), MF.getSubtarget().getRegisterInfo());
  }
  return false;
}

bool RISCVRegisterPressure::trySinkLoad(MachineFunction &MF, RegisterPressureHotSpot &P, std::vector<Register> &Regs, Register TargetReg) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  SlotIndex BeginIndex = LIS->getInstructionIndex(*P.BeginPos), EndIndex = LIS->getInstructionIndex(*P.EndPos);

  LLVM_DEBUG(dbgs() << "RP: trySinkLoad: TargetReg:" << printReg(TargetReg, TRI, 0, &MRI) << "\n");
  LLVM_DEBUG(dbgs() << "RP: trySinkLoad: IndexRange: (" << BeginIndex << ", " << EndIndex << ")\n");

  llvm::MachineBasicBlock::iterator MLI;
  MachineBasicBlock *MB = LIS->getInstructionFromIndex(BeginIndex)->getParent();
  //SmallVector<Register, 16> UsedRegs;

  LLVM_DEBUG(dbgs() << "RP: MII: new\n");
  // for (auto MII = MB->begin(); MII != MB->end(); MII++) {
  //   for (const MachineOperand & MO : MII->operands()) {
  //     if (!MO.isReg() || MO.getReg() == 0)
  //       continue;
  //     Register Reg = MO.getReg();
  //     if (!is_contained(UsedRegs, Reg))
  //       UsedRegs.push_back(Reg);
  //   }
  //   if (MII->definesRegister(TargetReg, MF.getSubtarget().getRegisterInfo()))
  //     LLVM_DEBUG(dbgs() << "RP: MII DEFINE REG:" << *MII << "\n");
  //   else if (MII->readsVirtualRegister(TargetReg))
  //     LLVM_DEBUG(dbgs() << "RP: MII READ REG:" << *MII << "\n");
  // }
  


  //for (auto MII = MRI.reg_instr_nodbg_begin(TargetReg); MII != MRI.reg_instr_nodbg_end(); MII++) {
  bool prevIsVLoad = false;
  bool hasMLI = false;
  for (auto MII = MB->begin(); MII != MB->end(); MII++) {
    SlotIndex MIIIndex = LIS->getInstructionIndex(*MII);
    LLVM_DEBUG(dbgs() << "RP: MII:" << *MII << "\n");
    LLVM_DEBUG(dbgs() << "RP: MII Opcode:" << MII->getOpcode() << "\n");
    if (MII->definesRegister(TargetReg, MF.getSubtarget().getRegisterInfo()) && isVLoad(*MII)) {
      if (isVLoad(*MII)) {
        LLVM_DEBUG(dbgs() << "RP: VLOAD:" << *MII << "\n");
        MLI = MII;
        prevIsVLoad = true;
        hasMLI = true;
      } else {
        prevIsVLoad = false;
        hasMLI = false;
      }
    } else if(MII->readsVirtualRegister(TargetReg)) {
      LLVM_DEBUG(dbgs() << "RP: check Sink or ReLoad\n");

      if (prevIsVLoad && (MIIIndex > BeginIndex)) {
        // Sink Load
        // LOAD <- MLI
        // ...
        // USE  <- MII
        // ...
        // USE
        // MachineBasicBlock::iterator WhereIter = (MIIIndex < EndIndex) ? LIS->getInstructionFromIndex(MIIIndex)->getIterator() : LIS->getInstructionFromIndex(EndIndex)->getIterator();
        LLVM_DEBUG(dbgs() << "RP: do sink load\n");
        MachineBasicBlock::iterator WhereIter = LIS->getInstructionFromIndex(MIIIndex)->getIterator();
        auto *MB = MII->getParent();
        LLVM_DEBUG(dbgs() << "RP: MOVE " << *MLI << "TO " << *MII << "\n");
        MB->splice(WhereIter, MB, MLI->getIterator());

        LIS->handleMove(*MLI);
        return true;
      } else if ((MIIIndex > BeginIndex) && hasMLI) {
        // Reload
        // LOAD <- MLI
        // ...
        // USE
        // ...
        // USE  < - MII
        const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
        const RISCVInstrInfo *TII = ST.getInstrInfo();
        auto Opcode = MLI->getOpcode();
        DebugLoc DL;
        for (unsigned i = 0; i < MLI->getNumOperands(); i++) {
          LLVM_DEBUG(dbgs() << "RP: operand: " << i << " " << MLI->getOperand(i) << "\n");
        }
        LLVM_DEBUG(dbgs() << "RP: do reload\n");

        //auto newReg = MRI.cloneVirtualRegister(MLI->getOperand(0).getReg());
        auto newMI = BuildMI(*MII->getParent(), MII->getIterator(), DL, TII->get(Opcode), MLI->getOperand(0).getReg());

        // auto newMI = BuildMI(*MII->getParent(), MII->getIterator(), DL, TII->get(Opcode), newReg);
        newMI.addUse(MLI->getOperand(1).getReg());
        LIS->InsertMachineInstrInMaps(*newMI.getInstr());

        // LIS->getRegUnit(newReg);
        LIS->removeInterval(MLI->getOperand(0).getReg());
        LIS->removeInterval(MLI->getOperand(1).getReg());

        //LIS->removeRegUnit(MLI->getOperand(1).getReg());
        LIS->getInterval(MLI->getOperand(0).getReg());
        LIS->getInterval(MLI->getOperand(1).getReg());
        //LIS->getInterval(MLI->getOperand(1).getReg());
        //LIS->createAndComputeVirtRegInterval(newReg);
        //LLVM_DEBUG(dbgs() << "RP: newMI" << *newMI << "\n");
        //newMI->addOperand(
        // LIS->repairIntervalsInRange(MB, MB->begin(), MB->end(), UsedRegs);
        //LIS->releaseMemory();
        //LIS->runOnMachineFunction(MF);
        return true;
      } else {
        prevIsVLoad = false;
      }
    }
  }

  // iterate over all the def_instr and find a LOAD instr
  /*
  for (auto MII = MRI.reg_instr_nodbg_begin(TargetReg); MII != MRI.reg_instr_nodbg_end(); MII++) {
    auto isLoad = isVLoad(*MII);
    SlotIndex MIIIndex = LIS->getInstructionIndex(*MII);
    LLVM_DEBUG(dbgs() << "RP: trySinkLoad: "<< *MII << isLoad << "\n");

    if (!isLoad)
      continue;

    auto ToIter = LIS->getInstructionFromIndex(EndIndex)->getIterator();
    auto nextMII = std::next(MII);
    if (!nextMII.atEnd() && LIS->getInstructionIndex(*nextMII) < EndIndex) {
      ToIter = (*nextMII).getIterator();
    }

    LLVM_DEBUG(dbgs() << "RP: trySinkLoad: move " << *MII << "to " << *ToIter);
    auto *MB = MII->getParent();
    MB->splice(ToIter, MB, MII->getIterator());

    LIS->handleMove(*MII);
    return true;
  }
  */

  return false;
}

bool RISCVRegisterPressure::runOnMachineFunction(MachineFunction &MF) {
  if (!enableRegisterPressureOpt())
    return false;
  //LLVM_DEBUG(dbgs() << "RP:Entering RegisterPressure for " << MF.getName() << "\n");
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // Once per function, before any loop handling or early return.
  //reportFunctionPressure(MF, LIS);

  bool change = false;
  if (MLI->empty()) {
    return false;
  }
  for (auto  ML : *MLI) {		// for all loop in function
    //auto LoopID = getLoopID(ML);  
    //auto IsVectorized = getOptionalBoolLoopAttribute(LoopID, "llvm.loop.isvectorized");

    // dbgs() << "getNumRegPressureSets" << TRI->getNumRegPressureSets() << '\n';
    // for (unsigned i = 0, e = TRI->getNumRegPressureSets(); i < e; ++i) {
    //   dbgs() << "i = " << i  << ":" << TRI->getRegPressureSetName(i)  << '\n';
    // }
    // static const char *PressureNameTable[] = {
    // "GPRC_and_PGPR", 0
    // "GPRX0",
    // "SP",
    // "VCSR",
    // "FPR32C",
    // "GPRC",
    // "SR07",
    // "VMV0",
    // "PGPR",
    // "GPRC_with_SR07",
    // "GPRTC",
    // "PGPR_with_GPRC",
    // "VRM8NoV0",
    // "FPR16",
    // "VM",  14
    // "GPR", 15
    // };
    

    RegisterPressureHotSpot P;
    bool changed = true;
    
    while (changed && hasHotSpotInLoop(MF, ML, P)) {
      std::vector<Register> LiveVRegs;
      SlotIndex BeginIndex = LIS->getInstructionIndex(*P.BeginPos), EndIndex = LIS->getInstructionIndex(*P.EndPos);
      LLVM_DEBUG(dbgs() << "RP: FindHotSpot " << "\n");
      LLVM_DEBUG(dbgs() << "RP: P.BeginPos " << *P.BeginPos << "SlotIndex: " << BeginIndex << "\n");
      LLVM_DEBUG(dbgs() << "RP: P.EndPos " << *P.EndPos << "SlotIndex: " << EndIndex << "\n");

      auto &MRI = MF.getRegInfo();
      //LIS->print(dbgs(), MF.getFunction().getParent());
      LLVM_DEBUG(dbgs() << "RP: getNumVirtRegs " << MRI.getNumVirtRegs() << "\n");
      for (unsigned i = 0; i < MRI.getNumVirtRegs(); ++i) {
        Register Reg = Register::index2VirtReg(i);
        auto *RC = MRI.getRegClass(Reg);
        auto *Pset = TRI->getRegClassPressureSets(RC);
        if (*Pset == 14) {
          LiveInterval &LI = LIS->getInterval(Reg);
          if (LI.overlaps(BeginIndex, EndIndex)) {
            LiveVRegs.push_back(Reg);
            LLVM_DEBUG(dbgs() << "RP: push to LiveVRegs: " << LI << "\n");
          }
        }
        LLVM_DEBUG(dbgs() << "Register: " << printReg(Reg, TRI, 0, &MRI) << " RegClass: " << TRI->getRegClassName(RC) << " Pset: " << *Pset << " PSName: " << TRI->getRegPressureSetName(*Pset) << "\n");
      }
      // for (auto Reg: LiveVRegs) {
      //   LLVM_DEBUG(dbgs() << "Register: " << printReg(Reg, TRI, 0, &MRI) << " RegClass: " << TRI->getRegClassName(MRI.getRegClass(Reg)) << "\n");
      // }
      // for (unsigned i = 0; i < TRI->getNumRegClasses(); i++) {
      //   LLVM_DEBUG(dbgs() << i << " " << TRI->getRegClassName(TRI->getRegClass(i)) << "\n");
      // }
      auto MII = P.BeginPos.getInstrIterator();
      auto MII_end = P.EndPos.getInstrIterator();
      // pair: active, score
      std::vector<std::pair<int, int>> scores;
      for (auto Reg: LiveVRegs) {
        LLVM_DEBUG(dbgs() << "RP: " << printReg(Reg, TRI, 0, &MRI) << "\n");
        std::pair<int, int> score;
        score.first = 1;
        score.second = 0;
        scores.push_back(score);
      }
      for (; MII != MII_end; MII++) {
        for (size_t i = 0; i < LiveVRegs.size(); i++) {
          auto RW = (*MII).readsWritesVirtualRegister(LiveVRegs[i]);
          if (RW.first || RW.second) {
            scores[i].first = 0;
          } else if (scores[i].first) {
            scores[i].second += 1;
          }
        }
      }

      // get highest score Reg
      int maxID, maxScore = -1;
      for (size_t i = 0; i < LiveVRegs.size(); i++) {
        if (scores[i].second > maxScore) {
          maxID = i;
          maxScore = scores[i].second;
        }
        LLVM_DEBUG(dbgs() << printReg(LiveVRegs[i], TRI, 0, &MRI) << "scores: " << scores[i].second << "\n");
      }

      changed = false;

      changed |= trySinkLoad(MF, P, LiveVRegs, LiveVRegs[maxID]);
      change |= changed;
    }
  }
  //LLVM_DEBUG(dbgs() << "Exist RegisterPressure for " << MF.getName() << "\n");
  return change;
}

FunctionPass *llvm::createRISCVRegisterPressurePass() {
  return new RISCVRegisterPressure();
}


