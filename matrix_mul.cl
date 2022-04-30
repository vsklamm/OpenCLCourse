// TILE_H == TILE_W
__kernel void matrix_mul(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         uint N,
                         uint K,
                         uint M)
{
    size_t i = get_local_id(0);
    size_t j = get_local_id(1);

    size_t global_i = get_global_id(0);
    size_t global_j = get_global_id(1);

    local float local_buf_a[TILE_H][TILE_W];
    local float local_buf_b[TILE_W][TILE_H];

    float tsum = 0.0;
    for (int l = 0; l < K; l += TILE_W)
    {
        local_buf_a[i][j] = a[global_i * K + l + j];
        local_buf_b[j][i] = b[M * (l + i) + global_j];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int d = 0; d < TILE_W; ++d)
        {
            tsum += local_buf_a[i][d] * local_buf_b[j][d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_i * M + global_j] = tsum;
}