#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

const char *TEST_PASSED = "OK";
const char *TEST_FAILED = "FAILED";

float random_float()
{
    return (float)rand() / (float)(RAND_MAX);
}

/**
 * Just prints array
 */
void print_array(cl_float arr[], cl_uint N)
{
    for (cl_uint i = 0; i < N; ++i)
    {
        printf("%.2f ", arr[i]);
    }
    printf("\n\n");
}

void random_array(cl_float arr[], cl_uint N)
{
    for (cl_uint i = 0; i < N; ++i)
    {
        arr[i] = random_float();
    }
}

void test_solution(cl_float in_array[], cl_float prefix_sum[], cl_uint N)
{
    printf("Testing solution... ");
    cl_float *answer = (cl_float *)malloc(N * sizeof(cl_float));
    for (cl_uint i = 0; i < N; ++i)
    {
        answer[i] = 0;
    }
    answer[0] = in_array[0];
    for (cl_uint i = 1; i < N; ++i)
    {
        answer[i] = answer[i - 1] + in_array[i];
    }
    for (cl_uint i = 0; i < N; ++i)
    {
        if (fabs(answer[i] - prefix_sum[i]) > 0.0001) // min: > 0.00001
        {
            printf("%s\n", TEST_FAILED);
            return;
        }
    }
    printf("%s\n", TEST_PASSED);
    free(answer);
}
