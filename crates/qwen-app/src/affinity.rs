// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Afinidad de hilos SMT en Win32: restringe el proceso a un hilo lógico por
//! núcleo físico en sistemas con un solo grupo de procesadores para evitar
//! contención SMT del planificador de Windows.

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

    pub fn pin_one_thread_per_core(silent: bool) -> usize {
        unsafe {
            if GetActiveProcessorGroupCount() > 1 {
                if !silent {
                    eprintln!(
                        "Afinidad: varios grupos de procesadores; no se restringe a un hilo por núcleo"
                    );
                }
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
                let info = std::ptr::read_unaligned(
                    buf.as_ptr().add(off) as *const SystemLogicalProcessorInformationEx
                );
                if (info.size as usize)
                    < std::mem::size_of::<SystemLogicalProcessorInformationEx>()
                    || off + (info.size as usize) > len as usize
                {
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
            // La afinidad se fija siempre; `silent` solo calla el mensaje.
            let pinned = mask != 0
                && mask != proc_mask
                && SetProcessAffinityMask(proc_handle, mask) != 0;
            if pinned && !silent {
                eprintln!(
                    "Afinidad: un hilo lógico por núcleo físico ({cores} núcleos, máscara 0x{mask:x})"
                );
            }
            cores.max(1)
        }
    }
}

/// Fija la afinidad (solo en Windows) y devuelve los núcleos físicos. Con `silent` no imprime
/// el mensaje de afinidad.
#[cfg(target_os = "windows")]
pub fn pin_one_thread_per_core(silent: bool) -> usize {
    win32::pin_one_thread_per_core(silent)
}

/// Fuera de Windows no se fija afinidad. Devuelve los hilos lógicos, que coinciden con los núcleos físicos en ARM64 (sin SMT); en x86_64 con SMT son el doble.
#[cfg(not(target_os = "windows"))]
pub fn pin_one_thread_per_core(_silent: bool) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
