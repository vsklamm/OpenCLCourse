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
void print_array(const char *name, float arr[], int rows, int columns)
{
    printf("%s %s:\n", "matrix", name);
    for (int i = 0; i < rows * columns; ++i)
    {
        printf("%.2f ", arr[i]);
    }
    printf("\n\n");
}

/**
 * Prints matrix in which elements are written sequentially in 1-dim array
 * [11, 12, 13, 21, 22, 23...]
 */
void print_matrix(const char *name, float arr[], int rows, int columns)
{
    printf("%s %s:\n", "matrix", name);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            printf("%.3f ", arr[i * columns + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Creates random matrix, elements are written sequentially in 1-dim array
 * [11, 12, 13, 21, 22, 23...]
 */
void random_matrix(float arr[], int rows, int columns)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            arr[i * columns + j] = random_float();
        }
    }
}

/**
 * Rotates matrix with memory reallocation
 * [rows x columns] -> [columns x rows]
 */
void rotate_matrix(float arr[], int rows, int columns)
{
    float *tmp = (float *)malloc(sizeof(float) * rows * columns);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            tmp[j * rows + i] = arr[i * columns + j];
        }
    }
    memcpy(arr, tmp, sizeof(float) * rows * columns);
    free(tmp);
}

// SOLUTIONS
/**
 * Naive solution for non-rotated matrices
 * [N x K] * [K x M]
 */
void naive(float res[], float a[], float b[], int N, int K, int M)
{
#pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            float sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += a[i * K + k] * b[k * M + j];
            }
            res[i * M + j] = sum;
        }
    }
}

/**
 * Naive solution for non-rotated matrices
 * [N x K] * [M x K]
 */
void naive_for_inverted(float res[], float a[], float b[], int N, int K, int M)
{
#pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            float sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += a[i * K + k] * b[j * K + k];
            }
            res[i * M + j] = sum;
        }
    }
}

void test_solution(float res[], float a[], float b[], int N, int K, int M)
{
    printf("Testing solution... ");
    float *answer = (cl_float *)malloc(N * M * sizeof(cl_float));
    naive(answer, a, b, N, K, M);
    for (int i = 0; i < N * M; ++i)
    {
        if (fabs(answer[i] - res[i]) > 0.0001) // min: > 0.00001
        {
            printf("%s\n", TEST_FAILED);
            return;
        }
    }
    printf("%s\n", TEST_PASSED);
    free(answer);
}
