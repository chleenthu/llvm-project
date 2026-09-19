//===-- ExpandPseudos.h - Expand Pseudos ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a pass that computes MachineUniformityInfo *before*
/// PHI elimination, so it can be safely used by later passes like register
/// pressure analysis while the function is still in SSA form.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EXPAND_PSEUDOS_H
#define LLVM_LIB_TARGET_EXPAND_PSEUDOS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class ExpandPseudos : public MachineFunctionPass {
  const TargetSubtargetInfo *STI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  DenseSet<Register> RegsToClearKillFlags;
  // Destinations of loads created by the remat transforms; sinking skips them.
  DenseSet<Register> RematRegs;

  using MIRegs = std::pair<MachineInstr *, SmallVector<Register, 2>>;

  using AllSuccsCache =
      SmallDenseMap<MachineBasicBlock *, SmallVector<MachineBasicBlock *, 4>>;

  using SeenDbgUser = PointerIntPair<MachineInstr *, 1>;

  using SinkItem = std::pair<MachineInstr *, MachineBasicBlock *>;

  SmallDenseMap<Register, TinyPtrVector<SeenDbgUser>> SeenDbgUsers;

  DenseSet<DebugVariable> SeenDbgVars;

  DenseMap<const MachineBasicBlock *, std::vector<unsigned>>
      CachedRegisterPressure;

public:
  static char ID;

  ExpandPseudos();

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;


private:
  MachineInstr *FindInSameSuccToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB,
                                      bool &BreakPHIEdge,
                                      AllSuccsCache &AllSuccessors);
  bool SinkInSameInstruction(MachineInstr &MI, bool &SawStore,
                       AllSuccsCache &AllSuccessors);
  bool SinkInSameInstructionGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, bool &SawStore,
    AllSuccsCache &AllSuccessors);
  bool FindFirstUseToSinkTo(MachineInstr &MI, AllSuccsCache &AllSuccessors);
  bool FindFirstUseToSinkToGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, AllSuccsCache &AllSuccessors);
  bool ProcessRedundantReload(MachineFunction &MF);
  bool ProcessRematLoads(MachineFunction &MF);
  void ProcessInSameBlock(MachineFunction &MF);
  void ProcessInSameAffine(MachineFunction &MF);
  bool runProcess(MachineFunction &MF);
  bool LowerCopy(MachineBasicBlock &MBB, MachineInstr &MI);
};

extern char &ExpandPseudosID;

} // end namespace llvm

#endif // LLVM_LIB_TARGET_EXPAND_PSEUDOS_H
