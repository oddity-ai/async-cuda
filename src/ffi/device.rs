use cpp::cpp;

use crate::device::DeviceId;
use crate::device::MemoryInfo;
use crate::ffi::result;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Post CUDA 11, NVIDIA has only appended to this struct
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CudaDeviceProp {
    /// ASCII string identifying the device.
    pub name: [u8; 256],

    /// 16-byte unique identifier.
    pub uuid: [u8; 16],

    /// 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms
    pub luid: [u8; 8],

    ///  LUID device node mask. Value is undefined on TCC and non-Windows platforms
    ///  original name: luidDeviceNodeMask
    pub luid_device_node_mask: u32,

    /// Total amount of global memory available on the device in bytes.
    /// original name: totalGlobalMem
    pub total_global_mem: usize,

    /// Maximum amount of shared memory available to a thread block in bytes.
    /// original name: sharedMemPerBlock
    pub shared_mem_per_block: usize,

    /// Maximum number of 32-bit registers available to a thread block.
    /// original name: regsPerBlock
    pub regs_per_block: i32,

    /// Warp size in threads.
    /// original name: warpSize
    pub warp_size: i32,

    /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cudaMallocPitch().
    /// original name: memPitch
    pub mem_pitch: usize,

    /// Maximum number of threads per block.
    /// original name: maxThreadsPerBlock
    pub max_threads_per_block: i32,

    /// Maximum size of each dimension of a block.
    /// original name: maxThreadsDim[3]
    pub max_threads_dim: [i32; 3],

    /// Maximum size of each dimension of a grid.
    /// original name: maxGridSize[3]
    pub max_grid_size: [i32; 3],

    /// Clock frequency in kilohertz.
    /// original name: clockRate
    pub clock_rate: i32,

    /// Total amount of constant memory available on the device in bytes.
    /// original name: totalConstMem
    pub total_const_mem: usize,

    /// Major revision number defining the device's compute capability.
    /// original name: major
    pub major: i32,

    /// Minor revision number defining the device's compute capability.
    /// original name: minor
    pub minor: i32,

    /// Alignment requirement; texture base addresses that are aligned to texture_alignment bytes do not need an offset applied to texture fetches.
    /// original name: textureAlignment
    pub texture_alignment: usize,

    /// Pitch alignment requirement for 2D texture references that are bound to pitched memory.
    /// original name: texturePitchAlignment
    pub texture_pitch_alignment: usize,

    /// 1 if the device can concurrently copy memory between host and device while executing a kernel, or 0 if not.
    /// original name: deviceOverlap
    pub device_overlap: i32,

    /// Number of multiprocessors on the device.
    /// original name: multiProcessorCount
    pub multi_processor_count: i32,

    /// 1 if there is a run time limit for kernels executed on the device, or 0 if not.
    /// original name: kernelExecTimeoutEnabled
    pub kernel_exec_timeout_enabled: i32,

    /// 1 if the device is an integrated (motherboard) GPU and 0 if it is a discrete (card) component.
    /// original name: integrated
    pub integrated: i32,

    /// 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not.
    /// original name: canMapHostMemory
    pub can_map_host_memory: i32,

    /// Compute mode that the device is currently in.
    /// original name: computeMode
    pub compute_mode: i32,

    /// Maximum 1D texture size.
    /// original name: maxTexture1D
    pub max_texture1d: i32,

    /// Maximum 1D mipmapped texture size.
    /// original name: maxTexture1DMipmap
    pub max_texture1d_mipmap: i32,

    /// Maximum 1D texture size for textures bound to linear memory.
    /// original name: maxTexture1DLinear
    pub max_texture1d_linear: i32,

    /// Maximum 2D texture dimensions.
    /// original name: maxTexture2D[2]
    pub max_texture2d: [i32; 2],

    /// Maximum 2D mipmapped texture dimensions.
    /// original name: maxTexture2DMipmap[2]
    pub max_texture2d_mipmap: [i32; 2],

    /// Maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
    /// original name: maxTexture2DLinear[3]
    pub max_texture2d_linear: [i32; 3],

    /// Maximum 2D texture dimensions if texture gather operations have to be performed.
    /// original name: maxTexture2DGather[2]
    pub max_texture2d_gather: [i32; 2],

    /// Maximum 3D texture dimensions.
    /// original name: maxTexture3D[3]
    pub max_texture3d: [i32; 3],

    /// Maximum alternate 3D texture dimensions.
    /// original name: maxTexture3DAlt[3]
    pub max_texture3d_alt: [i32; 3],

    /// Maximum cubemap texture width or height.
    /// original name: maxTextureCubemap
    pub max_texture_cubemap: i32,

    /// Maximum 1D layered texture dimensions.
    /// original name: maxTexture1DLayered[2]
    pub max_texture1d_layered: [i32; 2],

    /// Maximum 2D layered texture dimensions.
    /// original name: maxTexture2DLayered[3]
    pub max_texture2d_layered: [i32; 3],

    /// Maximum cubemap layered texture dimensions.
    /// original name: maxTextureCubemapLayered[2]
    pub max_texture_cubemap_layered: [i32; 2],

    /// Maximum 1D surface size.
    /// original name: maxSurface1D
    pub max_surface1d: i32,

    /// Maximum 2D surface dimensions.
    /// original name: maxSurface2D[2]
    pub max_surface2d: [i32; 2],

    /// Maximum 3D surface dimensions.
    /// original name: maxSurface3D[3]
    pub max_surface3d: [i32; 3],

    /// Maximum 1D layered surface dimensions.
    /// original name: maxSurface1DLayered[2]
    pub max_surface1d_layered: [i32; 2],

    /// Maximum 2D layered surface dimensions.
    /// original name: maxSurface2DLayered[3]
    pub max_surface2d_layered: [i32; 3],

    /// Maximum cubemap surface width or height.
    /// original name: maxSurfaceCubemap
    pub max_surface_cubemap: i32,

    /// Maximum cubemap layered surface dimensions.
    /// original name: maxSurfaceCubemapLayered[2]
    pub max_surface_cubemap_layered: [i32; 2],

    /// Alignment requirements for surfaces.
    /// original name: surfaceAlignment
    pub surface_alignment: usize,

    /// 1 if the device supports executing multiple kernels within the same context simultaneously, or 0 if not.
    /// original name: concurrentKernels
    pub concurrent_kernels: i32,

    /// 1 if the device has ECC support turned on, or 0 if not.
    /// original name: ECCEnabled
    pub ecc_enabled: i32,

    /// PCI bus identifier of the device.
    /// original name: pciBusID
    pub pci_bus_id: i32,

    /// PCI device (sometimes called slot) identifier of the device.
    /// original name: pciDeviceID
    pub pci_device_id: i32,

    /// PCI domain identifier of the device.
    /// original name: pciDomainID
    pub pci_domain_id: i32,

    /// 1 if the device is using a TCC driver or 0 if not.
    /// original name: tccDriver
    pub tcc_driver: i32,

    /// 1 when the device can concurrently copy memory between host and device while executing a kernel.
    /// original name: asyncEngineCount
    pub async_engine_count: i32,

    /// 1 if the device shares a unified address space with the host and 0 otherwise.
    /// original name: unifiedAddressing
    pub unified_addressing: i32,

    /// Peak memory clock frequency in kilohertz.
    /// original name: memoryClockRate
    pub memory_clock_rate: i32,

    /// Memory bus width in bits.
    /// original name: memoryBusWidth
    pub memory_bus_width: i32,

    /// L2 cache size in bytes.
    /// original name: l2CacheSize
    pub l2_cache_size: i32,

    /// L2 cache's maximum persisting lines size in bytes.
    /// original name: persistingL2CacheMaxSize
    pub persisting_l2_cache_max_size: i32,

    /// Number of maximum resident threads per multiprocessor.
    /// original name: maxThreadsPerMultiProcessor
    pub max_threads_per_multi_processor: i32,

    /// 1 if the device supports stream priorities, or 0 if it is not supported.
    /// original name: streamPrioritiesSupported
    pub stream_priorities_supported: i32,

    /// 1 if the device supports caching of globals in L1 cache, or 0 if it is not supported.
    /// original name: globalL1CacheSupported
    pub global_l1_cache_supported: i32,

    /// 1 if the device supports caching of locals in L1 cache, or 0 if it is not supported.
    /// original name: localL1CacheSupported
    pub local_l1_cache_supported: i32,

    /// Maximum amount of shared memory available to a multiprocessor in bytes; this amount is shared by all thread blocks simultaneously resident on a multiprocessor.
    /// original name: sharedMemPerMultiprocessor
    pub shared_mem_per_multiprocessor: usize,

    /// Maximum number of 32-bit registers available to a multiprocessor; this number is shared by all thread blocks simultaneously resident on a multiprocessor.
    /// original name: regsPerMultiprocessor
    pub regs_per_multiprocessor: i32,

    /// 1 if the device supports allocating managed memory on this system, or 0 if it is not supported.
    /// original name: managedMemory
    pub managed_memory: i32,

    /// 1 if the device is on a multi-GPU board (e.g. Gemini cards), and 0 if not.
    /// original name: isMultiGpuBoard
    pub is_multi_gpu_board: i32,

    /// Unique identifier for a group of devices associated with the same board. Devices on the same multi-GPU board will share the same identifier.
    /// original name: multiGpuBoardGroupID
    pub multi_gpu_board_group_id: i32,

    /// Link between the device and the host supports native atomic operations
    /// original name: hostNativeAtomicSupported
    host_native_atomic_supported: i32,

    /// Ratio of single precision performance (in floating-point operations per second) to double precision performance.
    /// original name: singleToDoublePrecisionPerfRatio
    pub single_to_double_precision_perf_ratio: i32,

    /// 1 if the device supports coherently accessing pageable memory without calling cudaHostRegister on it, and 0 otherwise.
    /// original name: pageableMemoryAccess
    pub pageable_memory_access: i32,

    /// 1 if the device can coherently access managed memory concurrently with the CPU, and 0 otherwise.
    /// original name: concurrentManagedAccess
    pub concurrent_managed_access: i32,

    /// 1 if the device supports Compute Preemption, and 0 otherwise.
    /// original name: computePreemptionSupported
    pub compute_preemption_supported: i32,

    /// 1 if the device can access host registered memory at the same virtual address as the CPU, and 0 otherwise.
    /// original name: canUseHostPointerForRegisteredMem
    pub can_use_host_pointer_for_registered_mem: i32,

    /// 1 if the device supports launching cooperative kernels via cudaLaunchCooperativeKernel, and 0 otherwise.
    /// original name: cooperativeLaunch
    pub cooperative_launch: i32,

    /// 1 if the device supports launching cooperative kernels via cudaLaunchCooperativeKernelMultiDevice, and 0 otherwise.
    /// original name: cooperativeMultiDeviceLaunch
    pub cooperative_multi_device_launch: i32,

    /// Per device maximum shared memory per block usable by special opt in
    /// original name: sharedMemPerBlockOptin
    pub shared_mem_per_block_optin: usize,

    /// 1 if the device accesses pageable memory via the host's page tables, and 0 otherwise.
    /// original name: pageableMemoryAccessUsesHostPageTables
    pub pageable_memory_access_uses_host_page_tables: i32,

    /// 1 if the host can directly access managed memory on the device without migration, and 0 otherwise.
    /// original name: directManagedMemAccessFromHost
    pub direct_managed_mem_access_from_host: i32,

    /// Maximum number of thread blocks that can reside on a multiprocessor.
    /// original name: maxBlocksPerMultiProcessor
    pub max_blocks_per_multi_processor: i32,

    /// Maximum value of cudaAccessPolicyWindow::num_bytes.
    /// original name: accessPolicyMaxWindowSize
    pub access_policy_max_window_size: i32,

    /// Shared memory reserved by CUDA driver per block in bytes
    /// original name: reservedSharedMemPerBlock
    pub reserved_shared_mem_per_block: usize,

    /// Device supports host memory registration via ::cudaHostRegister.
    pub host_register_supported: i32,

    /// 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise
    pub sparse_cuda_array_supported: i32,

    /// Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as as read-only to the GPU
    pub host_register_read_only_supported: i32,

    /// External timeline semaphore interop is supported on the device
    pub timeline_semaphore_interop_supported: i32,

    /// 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise
    pub memory_pools_supported: i32,

    /// 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
    pub gpu_direct_rdma_supported: i32,

    /// Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum
    pub gpu_direct_rdma_flush_writes_options: u32,

    /// See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values
    pub gpu_direct_rdma_writes_ordering: i32,

    /// Bitmask of handle types supported with mempool-based IPC
    pub memory_pool_supported_handle_types: u32,

    /// 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    pub deferred_mapping_cuda_array_supported: i32,

    /// Device supports IPC Events.
    pub ipc_event_supported: i32,

    /// Indicates device supports cluster launch
    pub cluster_launch: i32,

    /// Indicates device supports unified pointers
    pub unified_function_pointers: i32,

    /// Reserved for future use
    pub reserved2: [i32; 2],

    /// Reserved for future use
    pub reserved1: i32,

    ///  Reserved for future use
    pub reserved: [i32; 60],
}

impl Default for CudaDeviceProp {
    fn default() -> Self {
        CudaDeviceProp {
            name: [0; 256],
            uuid: [0; 16],
            luid: [0; 8],
            luid_device_node_mask: 0,
            total_global_mem: 0,
            shared_mem_per_block: 0,
            regs_per_block: 0,
            warp_size: 0,
            mem_pitch: 0,
            max_threads_per_block: 0,
            max_threads_dim: [0; 3],
            max_grid_size: [0; 3],
            clock_rate: 0,
            total_const_mem: 0,
            major: 0,
            minor: 0,
            texture_alignment: 0,
            texture_pitch_alignment: 0,
            device_overlap: 0,
            multi_processor_count: 0,
            kernel_exec_timeout_enabled: 0,
            integrated: 0,
            can_map_host_memory: 0,
            compute_mode: 0,
            max_texture1d: 0,
            max_texture1d_mipmap: 0,
            max_texture1d_linear: 0,
            max_texture2d: [0; 2],
            max_texture2d_mipmap: [0; 2],
            max_texture2d_linear: [0; 3],
            max_texture2d_gather: [0; 2],
            max_texture3d: [0; 3],
            max_texture3d_alt: [0; 3],
            max_texture_cubemap: 0,
            max_texture1d_layered: [0; 2],
            max_texture2d_layered: [0; 3],
            max_texture_cubemap_layered: [0; 2],
            max_surface1d: 0,
            max_surface2d: [0; 2],
            max_surface3d: [0; 3],
            max_surface1d_layered: [0; 2],
            max_surface2d_layered: [0; 3],
            max_surface_cubemap: 0,
            max_surface_cubemap_layered: [0; 2],
            surface_alignment: 0,
            concurrent_kernels: 0,
            ecc_enabled: 0,
            pci_bus_id: 0,
            pci_device_id: 0,
            pci_domain_id: 0,
            tcc_driver: 0,
            async_engine_count: 0,
            unified_addressing: 0,
            memory_clock_rate: 0,
            memory_bus_width: 0,
            l2_cache_size: 0,
            persisting_l2_cache_max_size: 0,
            max_threads_per_multi_processor: 0,
            stream_priorities_supported: 0,
            global_l1_cache_supported: 0,
            local_l1_cache_supported: 0,
            shared_mem_per_multiprocessor: 0,
            regs_per_multiprocessor: 0,
            managed_memory: 0,
            is_multi_gpu_board: 0,
            multi_gpu_board_group_id: 0,
            single_to_double_precision_perf_ratio: 0,
            pageable_memory_access: 0,
            concurrent_managed_access: 0,
            compute_preemption_supported: 0,
            can_use_host_pointer_for_registered_mem: 0,
            cooperative_launch: 0,
            cooperative_multi_device_launch: 0,
            pageable_memory_access_uses_host_page_tables: 0,
            direct_managed_mem_access_from_host: 0,
            max_blocks_per_multi_processor: 0,
            access_policy_max_window_size: 0,
            host_native_atomic_supported: 0,
            reserved_shared_mem_per_block: 0,
            shared_mem_per_block_optin: 0,
            host_register_supported: 0,
            sparse_cuda_array_supported: 0,
            host_register_read_only_supported: 0,
            timeline_semaphore_interop_supported: 0,
            memory_pools_supported: 0,
            gpu_direct_rdma_supported: 0,
            gpu_direct_rdma_flush_writes_options: 0,
            gpu_direct_rdma_writes_ordering: 0,
            memory_pool_supported_handle_types: 0,
            deferred_mapping_cuda_array_supported: 0,
            ipc_event_supported: 0,
            cluster_launch: 0,
            unified_function_pointers: 0,
            reserved2: [0; 2],
            reserved1: 0,
            reserved: [0; 60],
        }
    }
}

/// Synchronous implementation of [`crate::num_devices`].
///
/// Refer to [`crate::num_devices`] for documentation.
pub fn num_devices() -> Result<usize> {
    let mut num = 0_i32;
    let num_ptr = std::ptr::addr_of_mut!(num);
    let ret = cpp!(unsafe [
        num_ptr as "std::int32_t*"
    ] -> i32 as "std::int32_t" {
        return cudaGetDeviceCount(num_ptr);
    });

    result!(ret, num as usize)
}

/// Synchronous implementation of [`crate::Device`].
///
/// Refer to [`crate::Device`] for documentation.
pub struct Device;

impl Device {
    #[inline]
    pub fn get() -> Result<DeviceId> {
        let mut id: i32 = 0;
        let id_ptr = std::ptr::addr_of_mut!(id);
        let ret = cpp!(unsafe [
            id_ptr as "int*"
        ] -> i32 as "int" {
            return cudaGetDevice(id_ptr);
        });
        result!(ret, id)
    }

    #[inline(always)]
    pub fn get_or_panic() -> DeviceId {
        Device::get().unwrap_or_else(|err| panic!("failed to get device: {err}"))
    }

    #[inline]
    pub fn set(id: DeviceId) -> Result<()> {
        let ret = cpp!(unsafe [
            id as "int"
        ] -> i32 as "int" {
            return cudaSetDevice(id);
        });
        result!(ret)
    }

    #[inline]
    pub fn get_properties(id: DeviceId) -> Result<CudaDeviceProp> {
        let mut props = CudaDeviceProp::default();
        let props_len = std::mem::size_of::<CudaDeviceProp>();
        cpp! {{
            #include <cuda_runtime_api.h>
            #include <cstring>
            #include <algorithm>
        }}
        let props_ptr = std::ptr::addr_of_mut!(props);

        let ret = cpp!(unsafe [
            id as "int",
            props_len as "uint64_t",
            props_ptr as "cudaDeviceProp*"
        ] -> i32 as "int" {
            cudaDeviceProp prop;
            const auto ret = cudaGetDeviceProperties(&prop, id);

            // Don't copy past the end, just in case cudaDeviceProp grows
            memcpy(props_ptr, &prop, std::min(size_t(props_len), sizeof(prop)));
            return ret;
        });

        result!(ret, props)
    }

    #[inline(always)]
    pub fn set_or_panic(id: DeviceId) {
        Device::set(id).unwrap_or_else(|err| panic!("failed to set device {id}: {err}"));
    }

    pub fn synchronize() -> Result<()> {
        let ret = cpp!(unsafe [] -> i32 as "std::int32_t" {
            return cudaDeviceSynchronize();
        });
        result!(ret)
    }

    pub fn memory_info() -> Result<MemoryInfo> {
        let mut free: usize = 0;
        let free_ptr = std::ptr::addr_of_mut!(free);
        let mut total: usize = 0;
        let total_ptr = std::ptr::addr_of_mut!(total);

        let ret = cpp!(unsafe [
            free_ptr as "std::size_t*",
            total_ptr as "std::size_t*"
        ] -> i32 as "std::int32_t" {
            return cudaMemGetInfo(free_ptr, total_ptr);
        });
        result!(ret, MemoryInfo { free, total })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_devices() {
        assert!(matches!(num_devices(), Ok(num) if num > 0));
    }

    #[test]
    fn test_get_device() {
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_set_device() {
        assert!(Device::set(0).is_ok());
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_synchronize() {
        assert!(Device::synchronize().is_ok());
    }

    #[test]
    fn test_device_properties() {
        let props = Device::get_properties(0).expect("Get properties failed");
        eprintln!("{:?}", props);

        // check that its reasonable ascii
        // > 32 (space)
        // <= 126 (tilde)
        assert!(props
            .name
            .iter()
            .all(|c| *c == 0 || (*c >= 32 && *c <= 126)));
        assert!(props.total_global_mem > 0);
        assert!(props.shared_mem_per_block > 0);
        assert!(props.max_threads_dim[0] > 0);
        assert!(props.max_threads_dim[1] > 0);
        assert!(props.max_threads_dim[2] > 0);
        assert!(props.max_grid_size[0] > 0);
        assert!(props.max_grid_size[1] > 0);
        assert!(props.max_grid_size[2] > 0);
        assert!(props.surface_alignment > 0);
        assert!(props.memory_clock_rate > 0);
        assert!(props.memory_bus_width > 0);
        assert!(props.l2_cache_size > 0);
        assert!(props.l2_cache_size > 0);
        assert!(props.multi_processor_count > 0);
    }

    #[test]
    fn test_memory_info() {
        let memory_info = Device::memory_info().unwrap();
        assert!(memory_info.free > 0);
        assert!(memory_info.total > 0);
    }
}
