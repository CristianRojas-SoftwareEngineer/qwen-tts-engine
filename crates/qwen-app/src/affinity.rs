// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Afinidad de hilos SMT en Win32: restringe el proceso a un hilo lógico por
//! núcleo físico en sistemas con un solo grupo de procesadores para evitar
//! contención SMT del planificador de Windows (`INTERFACE_CONTRACT.md` §4.2).

#[cfg(target_os = "windows")]
mod win32 {
    const RELATION_PROCESSOR_CORE: u32 = 0;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct ProcessorRelationship {
        flags: u8,
        efficiency_class: u8,
        reserved: [u8; 20],
        group_count: u16,
        group_mask: [GroupAffinity; 1],
    }

    #[repr(C)]
    struct SystemLogicalProcessorInformationEx {
        relationship: u32,
        size: u32,
        processor: ProcessorRelationship,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetActiveProcessorGroupCount() -> u16;
        fn GetLogicalProcessorInformationEx(
            relationship_type: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
        fn GetCurrentProcess() -> isize;
        fn GetProcessAffinityMask(
            h_process: isize,
            lp_process_affinity_mask: *mut usize,
            lp_system_affinity_mask: *mut usize,
        ) -> i32;
        fn SetProcessAffinityMask(h_process: isize, dw_process_affinity_mask: usize) -> i32;
    }

    pub fn pin_one_thread_per_core() -> usize {
        unsafe {
            if GetActiveProcessorGroupCount() > 1 {
                eprintln!(
                    "Afinidad: varios grupos de procesadores; no se restringe a un hilo por núcleo"
                );
                return std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
            }
            let mut len: u32 = 0;
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                std::ptr::null_mut(),
                &mut len,
            );
            if len == 0 {
                return std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
            }
            let mut buf = vec![0u8; len as usize];
            if GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                buf.as_mut_ptr(),
                &mut len,
            ) == 0
            {
                return std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
            }

            let mut mask: usize = 0;
            let mut cores: usize = 0;
            let mut off: usize = 0;
            while off + std::mem::size_of::<SystemLogicalProcessorInformationEx>() <= len as usize {
                let info = &*(buf.as_ptr().add(off)
                    as *const SystemLogicalProcessorInformationEx);
                if info.size == 0 {
                    break;
                }
                if info.relationship == RELATION_PROCESSOR_CORE && info.processor.group_count > 0 {
                    let ga = info.processor.group_mask[0];
                    if ga.group == 0 && ga.mask != 0 {
                        mask |= ga.mask & ga.mask.wrapping_neg();
                        cores += 1;
                    }
                }
                off += info.size as usize;
            }

            let proc_handle = GetCurrentProcess();
            let mut proc_mask: usize = 0;
            let mut sys_mask: usize = 0;
            if GetProcessAffinityMask(proc_handle, &mut proc_mask, &mut sys_mask) == 0 {
                return cores.max(1);
            }
            mask &= proc_mask;
            if mask != 0 && mask != proc_mask && SetProcessAffinityMask(proc_handle, mask) != 0 {
                eprintln!(
                    "Afinidad: un hilo lógico por núcleo físico ({cores} núcleos, máscara 0x{mask:x})"
                );
            }
            cores.max(1)
        }
    }
}

pub fn pin_one_thread_per_core() -> usize {
    #[cfg(target_os = "windows")]
    {
        win32::pin_one_thread_per_core()
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}
