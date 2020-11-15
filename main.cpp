#pragma GCC optimize(3)

#include <iostream>
#include <chrono>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cblas.h>
#include <omp.h>

#define timePrint  std::cout\
<< "Slow calculations took "\
<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "μs ≈ "\
<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms ≈ "\
<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s.\n";

using namespace std;

struct matrix {

public:

    int column = 0;
    int row = 0;
    float **a2;

    matrix(int row, int column) {
        this->row = row;
        this->column = column;
        a2 = new float *[row];
        for (int i = 0; i < row; i++)
            a2[i] = new float[column];
    }


};




float **calculate(matrix m, matrix n) {
    int lengthM = m.row;
    int widthM = m.column;
    int lengthN = n.row;
    int widthN = n.column;
    matrix x(m.row, n.column);

    if (widthM == lengthN) {
        for (int i = 0; i < lengthM; ++i) {
            for (int j = 0; j < widthN; ++j) {
                for (int k = 0; k < widthM; ++k) {
                    x.a2[i][j] += m.a2[i][k] * n.a2[k][j];

                }
            }
        }


    } else{
        cout<<"The size of the input matrices are wrong";
        exit(0);
    }


    return x.a2;
}

float **calculate2(matrix m, matrix n) {
    int lengthM = m.row;
    int widthM = m.column;
    int lengthN = n.row;
    int widthN = n.column;
    matrix x(m.row, n.column);

    float s = 0;
    if (widthM == lengthN) {
//#pragma omp parallel for num_threads(4)
        for (int i = 0; i < lengthM; ++i) {
            for (int k = 0; k < widthM; ++k) {
                s = m.a2[i][k];
                for (int j = 0; j < widthN; ++j) {
                    //          omp_set_nest_lock(&lock);
                    x.a2[i][j] += m.a2[i][k] * n.a2[k][j];
                    //         omp_unset_nest_lock(&lock);

                }
            }
        }

    }

    return x.a2;
}


float simd_dot(const float *x, const float *y, const long &len) {
    float inner_prod = 0.0f;
    __m256 X, Y; //声明两个512位专用寄存器的变量
    __m256 acc = _mm256_setzero_ps();// 声明一个存放在SSE的512位专用寄存器的变量，用来存放X+Y的结果，初始值为0
    float temp[8];//存放中间变量的参数
    long i;
    // #pragma omp parallel

    for (i = 0; i + 8 < len; i += 8) {//512位专用寄存器，一次性可以处理16组32位变量的运算
        X = _mm256_loadu_ps(x + i); // 将x加载到X（由于512位可以存放16个32位数据，所以默认一次加载连续的16个参数）
        Y = _mm256_loadu_ps(y + i);//同上
        acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));//x*y，每轮的x1*y1求和，x2*y2求和，x3*y3求和，x4*y4求和,最终产生的16个和，放在acc的512位寄存器中。
    }
    _mm256_storeu_ps(&temp[0], acc); // 将acc中的16个32位的数据加载进内存
    inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];//点乘求和

    for (; i < len; ++i) {
        inner_prod += x[i] * y[i];//继续累加小尾巴的乘积
    }
    return inner_prod;
}

float **calculate3(matrix m, matrix n) {
    int lengthM = m.row;
    int widthM = m.column;
    int lengthN = n.row;
    int widthN = n.column;
    matrix x(m.row, n.column);

    if (widthM == lengthN) {
   //     #pragma omp parallel for num_threads(8)

        for (int i = 0; i < lengthM; ++i) {
            for (int j = 0; j < widthN; ++j) {
                x.a2[i][j] = simd_dot(m.a2[i], n.a2[j], m.column);

            }
        }

    }

    return x.a2;
}



int main() {
    std::cout << "Hello, World!" << std::endl;


    matrix m(1000, 1000);
    int p = 0;
    for (int i = 0; i < m.row; i++) {
        for (int j = 0; j < m.column; j++) {
            m.a2[i][j] = (p++);
        }
    }


    matrix n(1000, 1000);
    for (int i = 0; i < n.row; i++) {
        for (int j = 0; j < n.column; j++) {
            n.a2[i][j] = 0;
        }
    }
    n.a2[0][0] = 1;

    int len = 10000;
    int wid = 10000;
    float *x = new float[len * wid];
    float *y = new float[len * wid];
    float c[len * wid];


    auto start = std::chrono::steady_clock::now();
    float **result = calculate(m, n);
    auto end = std::chrono::steady_clock::now();
    timePrint

    start = std::chrono::steady_clock::now();
    result = calculate2(m, n);
    end = std::chrono::steady_clock::now();
    timePrint

    start = std::chrono::steady_clock::now();
    result = calculate3(m, n);
    end = std::chrono::steady_clock::now();
    timePrint


    start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10000, 10000, 10000, 10000, x, 10000, y, 10000, 10000, c, 10000);
    end = std::chrono::steady_clock::now();
    timePrint



    for (int i = 0; i < m.row; i++) {
        for (int j = 0; j < n.column; j++) {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }




    return 0;
}
