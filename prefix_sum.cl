__kernel void prefix_sum(__global const float *in_array,
                         __global float *res_array,
                         uint n)
{
    int local_i = get_local_id(0);
    int local_i2 = local_i << 1;
    __local float buff[MAX_WORK_GR];

    buff[local_i2] = in_array[local_i2];
    buff[local_i2 + 1] = in_array[local_i2 + 1];

    int offset = 1;
    for (int l = n >> 1; l > 0; l >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_i < l)
        {
            int i = offset * (local_i2 + 1) - 1;
            int j = offset * (local_i2 + 2) - 1;
            buff[j] += buff[i];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i == 0)
    {
        buff[n - 1] = 0;
    }
    for (int l = 1; l < n; l <<= 1)
    {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_i < l)
        {
            int i = offset * (local_i2 + 1) - 1;
            int j = offset * (local_i2 + 2) - 1;
            float t = buff[i];
            buff[i] = buff[j];
            buff[j] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    res_array[local_i2] = buff[local_i2] + in_array[local_i2];
    res_array[local_i2 + 1] = buff[local_i2 + 1] + in_array[local_i2 + 1];
}