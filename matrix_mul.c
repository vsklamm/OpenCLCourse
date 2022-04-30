#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/opencl.h>
#include <omp.h>

#include "matrix.h"
#include "print_opencl_error.h"

#define ERR_CHECK(err_code, message, exit_label)                           \
    do                                                                     \
    {                                                                      \
        if ((err_code) != 0)                                               \
        {                                                                  \
            printf("Error at %s:%d. %s\n", __FILE__, __LINE__, (message)); \
            goto exit_label;                                               \
        }                                                                  \
    } while (0)

char *get_options_string(int TILE_H, int TILE_W)
{
    char options_tmp[50];
    int bytes = sprintf(options_tmp, "-D TILE_H=%d -D TILE_W=%d", TILE_H, TILE_W);
    char *options = (char *)malloc(bytes + 1);
    memcpy(options, options_tmp, bytes + 1);
    return options;
}

void print_device_info(cl_device_id def_device)
{
    size_t device_name_size;
    size_t cl_version_size;
    OPENCL_CHECK(clGetDeviceInfo(def_device, CL_DEVICE_NAME, 0, NULL, &device_name_size));
    char *device_name = (char *)malloc(device_name_size);
    OPENCL_CHECK(clGetDeviceInfo(def_device, CL_DEVICE_NAME, device_name_size, device_name, &device_name_size));
    OPENCL_CHECK(clGetDeviceInfo(def_device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &cl_version_size));
    char *value = (char *)malloc(cl_version_size);
    OPENCL_CHECK(clGetDeviceInfo(def_device, CL_DEVICE_OPENCL_C_VERSION, cl_version_size, value, NULL));
    printf("Target device: %s\n", device_name);
    printf("OpenCL C version: %s\n", value);
    free(device_name);
    free(value);
}

void get_all_platforms(cl_uint *num_platforms, cl_platform_id **platforms)
{
    CL_CHECK(clGetPlatformIDs(0, NULL, num_platforms), exit_platforms);
    *platforms = (cl_platform_id *)malloc(*num_platforms * sizeof(cl_platform_id));
    CL_CHECK(clGetPlatformIDs(*num_platforms, *platforms, NULL), exit_platforms);
    printf("Platforms found: %d\n", *num_platforms);
    if (*num_platforms)
    {
        return;
    }
exit_platforms:
    free(*platforms);
    exit(EXIT_FAILURE);
}

int main()
{
    srand(time(NULL));
    // task constants
    const int TILE_H = 32, TILE_W = 32;
    const char *code_file_name = "matrix_mul.cl";
    const char *kernel_name = "matrix_mul";
    const int N = 2048;
    const int K = 512;
    const int M = 1024;
    const size_t operations = N * K * M * 2LL;
    const size_t a_size = N * K * sizeof(cl_float);
    const size_t b_size = K * M * sizeof(cl_float);
    const size_t c_size = N * M * sizeof(cl_float);
    // error handling
    cl_int error_code;
    // platforms
    cl_uint num_platforms = 0;
    cl_platform_id *platforms = NULL;
    // devices
    cl_device_id *devices = NULL;
    cl_device_id def_device = NULL;

    get_all_platforms(&num_platforms, &platforms);

    // select default device
    cl_uint num_devices_gpu, num_devices_cpu;
    cl_device_id *devices_gpu = NULL, *devices_cpu = NULL;
    cl_device_id gpu = NULL, cpu = NULL;
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices_gpu);
        devices_gpu = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices_gpu);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices_gpu, devices_gpu, NULL);
        if (num_devices_gpu > 0)
        {
            gpu = devices_gpu[0];
            for (cl_uint j = 0; j < num_devices_gpu; ++j)
            {
                cl_bool has_unified_memory;
                clGetDeviceInfo(devices_gpu[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &has_unified_memory, NULL);
                if (has_unified_memory == CL_FALSE)
                {
                    def_device = devices_gpu[j];
                    break;
                }
            }
        }
        if (def_device)
        {
            break;
        }
        if (!gpu && !cpu)
        {
            CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices_cpu), exit_devices);
            devices_cpu = (cl_device_id *)malloc(num_devices_cpu * sizeof(cl_device_id));
            CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, num_devices_cpu, devices_cpu, NULL), exit_devices);
            if (num_devices_cpu > 0)
            {
                cpu = devices_cpu[0];
            }
        }
    }
    if (!def_device)
    {
        def_device = gpu ? gpu : cpu;
        devices = gpu ? devices_gpu : devices_cpu;
    }
    else
    {
        devices = devices_gpu;
    }
    print_device_info(def_device);

    // context
    cl_context context;
    cl_command_queue command_qu;

    context = clCreateContext(NULL, 1, devices, NULL, NULL, &error_code);
    CL_CHECK(error_code, exit_devices);
    command_qu = clCreateCommandQueue(context, def_device, CL_QUEUE_PROFILING_ENABLE, &error_code);
    CL_CHECK(error_code, exit_context);

    // reading source code
    char *source = NULL;
    size_t source_size = 0;
    FILE *f = fopen(code_file_name, "rb");
    if (f == NULL)
    {
        printf("File not found: %s\n", code_file_name);
        goto exit_context;
    }
    fseek(f, 0, SEEK_END);
    source_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    source = malloc(source_size + 1);
    fread(source, 1, source_size, f);
    fclose(f);
    source[source_size] = 0;
    const char *csource = source;
    // printf("%s\n", csource);

    char *options = get_options_string(TILE_H, TILE_W);
    const char *const_options = options;

    cl_program program = clCreateProgramWithSource(context, 1, &csource, &source_size, &error_code);
    CL_CHECK(error_code, exit_queue);
    error_code = clBuildProgram(program, 1, devices, const_options, NULL, NULL);
    if (error_code != CL_SUCCESS)
    {
        printf("Build program failed\n");
        size_t log_size;
        clGetProgramBuildInfo(program, def_device, CL_PROGRAM_BUILD_LOG, 0, 0, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, def_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
        CL_CHECK(error_code, exit_program);
    }
    printf("Program build success\n");

    // creating buffers for matrices
    cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code);
    CL_CHECK(error_code, exit_program);
    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY, a_size, NULL, &error_code);
    CL_CHECK(error_code, exit_kernel);
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY, b_size, NULL, &error_code);
    CL_CHECK(error_code, exit_mem_object_a);
    cl_mem buf_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_size, NULL, &error_code);
    CL_CHECK(error_code, exit_mem_object_b);

    cl_float *a = (cl_float *)malloc(a_size);
    cl_float *b = (cl_float *)malloc(b_size);
    cl_float *c = (cl_float *)malloc(c_size);

    random_matrix(a, N, K);
    random_matrix(b, K, M);
    // rotate_matrix(b, K, M);

    CL_CHECK(clEnqueueWriteBuffer(command_qu, buf_a, CL_FALSE, 0, a_size, a, 0, NULL, NULL), exit_all);
    CL_CHECK(clEnqueueWriteBuffer(command_qu, buf_b, CL_FALSE, 0, b_size, b, 0, NULL, NULL), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &K), exit_all);
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &M), exit_all);

    cl_uint work_dim = 2;
    const size_t global_work_size[] = {N, M};
    const size_t local_work_size[] = {TILE_H, TILE_W};
    cl_event run_event;
    CL_CHECK(clEnqueueNDRangeKernel(command_qu, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &run_event), exit_all);
    CL_CHECK(clEnqueueReadBuffer(command_qu, buf_c, CL_TRUE, 0, c_size, c, 0, NULL, NULL), exit_all);

    cl_ulong t_start = 0, t_end = 0;
    CL_CHECK(clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0), exit_all);
    CL_CHECK(clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0), exit_all);

    printf("Time: %lu ns. ~%.3f GFlops\n", t_end - t_start, (double)operations / (t_end - t_start));
    test_solution(c, a, b, N, K, M);
    // print_matrix("result", c, N, M);
    return EXIT_SUCCESS;

exit_all:
    free(c);
    free(b);
    free(a);
    clReleaseMemObject(buf_c);
exit_mem_object_b:
    clReleaseMemObject(buf_b);
exit_mem_object_a:
    clReleaseMemObject(buf_a);
exit_kernel:
    clReleaseKernel(kernel);
exit_program:
    clReleaseProgram(program);
exit_queue:
    clReleaseCommandQueue(command_qu);
    free(source);
    free(options);
exit_context:
    clReleaseContext(context);
exit_devices:
    free(platforms);
    free(devices);
    return EXIT_FAILURE;
}
