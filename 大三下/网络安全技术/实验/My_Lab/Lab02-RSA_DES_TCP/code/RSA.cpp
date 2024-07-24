#include <iostream>
#include <string>
#include <assert.h>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <bitset>
#include "RSA_handle.cpp"

using namespace std;

struct RSA_key_pair 
{
    unsigned long long public_key_e;
    unsigned long long private_key_d;
    unsigned long long n;
    string en;
};

// 计算 (a * b) % n 的乘法函数
unsigned long long multi(unsigned long long a, unsigned long long b, unsigned long long n)
{
    unsigned long long s = a;
    unsigned long long t = b;

    unsigned long long result = 0;
    a %= n;

    while (b > 0)
    {
        if (b % 2 == 1)
        {
            result = (result + a) % n;
        }
        a = (a * 2) % n;
        b /= 2;
    }

    // cout << "(a = " << s << " * b = " << t << ") % " << "n = " << n << " = " << result << endl;
    return result;
}

// 平方乘快速算法，计算 (a^m) mod n
unsigned long long square_mult(unsigned long long a, unsigned long long m, unsigned long long n)
{
    unsigned long long s = a;
    unsigned long long t = m;

    unsigned long long result = 1;
    a = a % n;

    while (m > 0)
    {
        // 如果指数是奇数，乘以当前的 a
        if (m % 2 == 1)
        {
            // result = (result * a) % n;
            result = multi(result, a, n);
        }
        // 将指数减半，底数平方
        m >>= 1;
        // a = (a * a) % n;
        a = multi(a, a, n);
    }
    // cout << "(a = " << s << " ^ b = " << t << ") % " << "n = " << n << " = " << result << endl;
    return result;
}

//平方乘算法(信安数基教材版)
unsigned long long square_and_multiply(unsigned long long base, unsigned long long exp, unsigned long long modulus)
{
    unsigned long long s = base;
    unsigned long long t = exp;

    unsigned long long result = 1;
    base = base % modulus;
    bitset<1024> binaryExponent(exp);

    for (int i = 0; i < binaryExponent.size(); i++)
    {
        // 如果指数的当前位是1，乘以当前的 base
        if (binaryExponent[i] == 1)
        {
            result = (result * base) % modulus;
        }
        // 底数平方
        base = (base * base) % modulus;
    }

    // cout << "(a = " << s << " ^ b = " << t << ") % " << "n = " << modulus << " = " << result << endl;
    return result;
}

// 模乘运算，即计算两个数的乘积然后取模
unsigned long long MulMod(unsigned long long a, unsigned long long b, unsigned long long n)
{
    return (a % n) * (b % n) % n;
    // return multi(a % n, b % n, n);
}

// 模幂运算，首先计算某数的若干次幂，然后对其结果进行取模运算。
unsigned long long PowMod(unsigned long long base, unsigned long long pow, unsigned long long n)
{
    unsigned long long a = base;// 变量a代表计算过程中未完成乘方运算的底数
    unsigned long long b = pow;// 变量b代表计算过程中未完成乘方运算的指数
    unsigned long long c = 1;// 变量c代表计算过程中运算中间变量，用于存储运算结果。

    while (b) 
    {
        // while (b % 2 == 0)
        while (!(b & 1)) // 在while循环中首先考虑未完成的乘方次数（b）是否为偶数
        {
            // b /= 2;
            b >>= 1;// 如果是偶数，b变为原来的1/2
            a = MulMod(a, a, n);// a变为a^2 %n
        }
        // 内层循环在未完成乘方次数b未奇数时跳出
        b--;// 将b-1
        c = MulMod(a, c, n);// 将a%n的值暂存在c
    }
    return c;
}

// Miller-Rabin算法判定素数性质
bool Miller_Rabin(unsigned long long n)
{
    // 当n小于等于1时，返回false
    if (n <= 1)
    {
        return false;
    }
    //2和3是素数
    if (n <= 3)
    {
        return n == 2 || n == 3;
    }
    // 如果n是偶数，则不是素数
    if (n % 2 == 0)
    {
        return false;
    }

    // 寻找 s 和 t，将 n-1分解成2^s * t，其中t是奇整数
    unsigned long long t = n - 1;
    unsigned long long s = 0;

    while (t % 2 == 0)
    {
        t /= 2;
        s++;
    }

    // 进行 k 次测试,k = n - 3
    // 随机选择一个在区间[2,n-2]中的整数b，计算 r0 = b^t mod n
    // 如果 b^t mod n = 1 或 n - 1，则n可能是素数
    // 如果存在一个r，0 <= r < s，使得r0^2 mod n=-1，n可能是素数
    // 如果上述两个条件都不成立，则n一定是合数

    for (int i = 0; i < 100; i++)
    {
        // 随机选择基数 b，范围在 [2, n-2]
        unsigned long long b = 2 + rand() % (n - 4);
        // long long b = 2 + i;

        // 使用 std::mt19937_64 作为随机数生成器
        // mt19937_64 rng(random_device{}());

        // 设置取值范围
        // unsigned long long n = static_cast<unsigned long long>(1) << 63;

        // 生成 [2, n-3] 范围内的随机数
        // uniform_int_distribution<unsigned long long> distribution(2, n - 3);
        // unsigned long long b = distribution(rng);

        // 计算 a^t % n
        unsigned long long r0 = square_mult(b, t, n);
        // unsigned long long r0 = square_and_multiply(b, t, n);

        unsigned long long r1;

        // 如果r0结果为 1或n-1 或 0，则通过检验，重选b继续下一次测试
        if (r0 == 1 || r0 == n - 1)
        {
            continue;
        }

        // 如果r0 !=1或r0 !=n-1，则计算r1= r0^2 mod n
        // 进行 s-1 次平方运算
        else
        {
            for (int r = 0; r < s; r++)
            {
                r1 = multi(r0, r0, n);

                // 如果 r1 = 1，则 n 一定是合数
                if (r1 == 1)
                {
                    return false;
                }

                // 如果 r1 = n-1，则通过检验，重选 b 继续下一次测试
                if (r1 == n - 1)
                {
                    break;
                }

                // 更新 r0 为 r1，准备下一轮的平方取模运算
                r0 = r1;
            }

            // 如果循环结束时，r1 仍然不等于 n-1，仍未找到 n 可能是素数的证据，则 n 不是素数
            if (r1 != n - 1)
            {
                return false;
            }
        }
    }
    // 测试中都没有发现 n 不是素数的证据，则 n 可能是素数
    return true;
}

// Rabin-Miller 素数测试
bool RabinMillerKnl(unsigned long long n)
{
    unsigned long long a;
    unsigned long long q = n - 1;
    unsigned long long k = 0;
    unsigned long long v;

    // 寻找 k 和 q，将 n-1分解成2^k * q，其中q是奇整数，k是非负整数
    // while (q % 2 == 0)
    while (!(q & 1))// 如果q是偶数
    {    
        q >>= 1;
        ++k;
    }

    // 随机数a满足 2 ≤ a < n − 1，即[2, n-2]
    a = 2 + rand() % (n - 3); 
    // 计算a^q % n，如果等于1，通过测试代表该数字可能是质数
    if (PowMod(a, q, n) == 1) 
    {
        return true;
    }
    // 循环检验 a^zq % n (z=2^j)的值，如果这个值等于n-1，则证明此数字可能是一个质数
    for (unsigned long long j = 0; j < k; j++)
    {
        unsigned long long z = 1;

        for (unsigned long long w = 0; w < j; w++)
        {
            z *= 2;
        }

        if (PowMod(a, z * q, n) == n - 1) 
        {
            return true;
        }
    }
    return false;
}

// 重复调用RabinMillerKnl()判别n是否是质数
bool RabinMiller_Isprime(unsigned long long n, int loop)
{
    for (int i = 0; i < loop; ++i) 
    {
        if (!RabinMillerKnl(n))
        {
            return false;
        }
    }
    return true;
}

// 质数生成函数
// 该函数首先生成一个确保最高位是一（确保足够大）的随机奇数，
// 然后，检验该奇数是否是质数，如该奇数不是质数，则重复该过程直到生成所需质数为止。
unsigned long long RandomPrime(char bits)
{
    unsigned long long base;

    static int rand_noise = 0;
    rand_noise++;
    srand((unsigned)time(NULL) + rand_noise);

    do 
    {
        base = (unsigned long)1 << (bits - 1); //保证最高位是 1
        base += rand() % base; //加上一个随机数
        base |= 1; //保证最低位是1，即保证是奇数
    } while (!RabinMiller_Isprime(base, 30)); //测试 30 次
    return base; //全部通过认为是质数
}

// 求最大公约数
unsigned long long gcd(unsigned long long p, unsigned long long q)
{
    unsigned long long a = max(p, q);
    unsigned long long b = min(p, q);
    int t;

    if (p == q)
    {
        return false; //两数相等,最大公约数就是本身
    }
    else
    {
        while (b)
        { // 辗转相除法，gcd(a,b)=gcd(b,a-qb)
            a = a % b;
            t = a;
            a = b;
            b = t;
        }
        return a;
    }
}

// 返回小于n且与n互质的正整数个数
unsigned long long Euler(unsigned long long n)
{
    unsigned long long res = n, a = n;
    for (unsigned long long i = 2; i * i <= a; ++i)
    {
        if (a % i == 0)
        {
            res = res / i * (i - 1);//先进行除法是防止中间数据的溢出
            while (a % i == 0) 
            {
                a /= i;
            }
        }
    }
    if (a > 1) 
    {
        res = res / a * (a - 1);
    }
    return res;
}

// 私钥生成
unsigned long long Euclid(unsigned long long e, unsigned long long t_n)
{
    unsigned long long Max = 0xffffffffffffffff - t_n;
    unsigned long long i = 1;
    unsigned long long d = 0;

    while (1)
    {
        if (((i * t_n) + 1) % e == 0)
        {
            return d = ((i * t_n) + 1) / e;
        }
        i++;
        unsigned long long Tmp = (i + 1) * t_n;
        if (Tmp > Max)
        {
            return 0;
        }
    }
    return 0;
}

// 服务器端生成RSA公私钥对
unsigned long long RSA_key_pair_gen(RSA_key_pair& key_pair) 
{
    // 生成公钥{e，n}
    unsigned long long Prime_p = RandomPrime(16);// 随机生成两个素数
    unsigned long long Prime_q = RandomPrime(16);
    unsigned long long n = Prime_p * Prime_q;
    unsigned long long f = (Prime_p - 1) * (Prime_q - 1);

    unsigned long long euler = Euler(n);
    unsigned long long e;

    while (1) 
    {
        e = rand() % 65536 + 1;
        e |= 1;
        if (gcd(e, f) != 1) 
        {
            break;
        }
    }

    cout << "RSA_publickey: " << "{ e: " << e << " , n: " << n << "} " << endl;

    // unsigned long long d = 0;
    // d = Euclid(e, f);
    
    // 生成私钥{d，n}
    unsigned long long max = 0xffffffffffffffff - euler;
    unsigned long long i = 1;
    unsigned long long d = 0;

    while (1)
    {
        if (((i * euler) + 1) % e == 0) 
        {
            d = ((i * euler) + 1) / e;
            break;
        }
        i++;
        unsigned long long temp = (i + 1) * euler;
        if (temp > max)
        {
            break;
        }
    }

    cout << "RSA_privatekey : " << "{ d: " << d << " , n: " << n << "} " << endl;
    cout << endl;

    // 如果循环结束后d值仍然为0表示密钥生成失败
    if (d == 0) 
    {
        return -1;
    }

    unsigned long long s = 0;
    unsigned long long t = n >> 1;
    while (t)
    {
        s++;
        t >>= 1;
    }

    key_pair.public_key_e = e;
    key_pair.private_key_d = d;
    key_pair.n = n;
    key_pair.en = to_string(e) + "," + to_string(n);
    return 0;
}


// 客户端使用服务器的公钥加密DES密钥，64位为需要分四次加密
// 使用服务器公钥，计算 C = (M^e) mod n
// 返回值是4个string类型的16位加密结果

string client_encry(string des_key, unsigned long long public_key_e, unsigned long long n)
{
    // 将8个字母的string类型DesKey转为64位int类型
    unsigned long long _des_key = 0;
    for (unsigned int i = 0; i < des_key.length(); ++i) 
    {
        _des_key += des_key[i];
        if (i != des_key.length() - 1) 
        {
            _des_key <<= 8;
        }
    }

    // 64位int拆成4份，每份16位
    unsigned short* p_res = (unsigned short*)&_des_key;
    unsigned short M[4];
    for (int i = 0; i < 4; ++i) 
    {
        M[i] = p_res[i];
    }

    string result;
    string result_out;
    // 对每一份执行加密函数，并将4个16位数字转成string，用逗号分隔
    for (int i = 3; i >= 0; --i) 
    {
        string temp = to_string(PowMod(M[i], public_key_e, n));
        result += temp;
        // result_out += temp;
        result += ',';
    }
    // cout << "result_out: " << result_out << endl;
    return result;
}

// 服务器使用私钥解密客户端发来的DES密钥
// 使用服务器私钥，计算 M = (C^d) mod n 
string server_decry(string key_info, unsigned long long private_key_d, unsigned long long n) 
{
    string des_key = "";
    int pos = 0;
    for (int i = 0; i < 4; i++)
    {
        string temp = "";
        for (; key_info[pos] != ','; ++pos) 
        {
            temp += key_info[pos];
        }
        ++pos;
        // cout << i << endl;
        // string temp = key_info.substr(2 * i, 2 * i + 1);
        // cout << i << endl;

        unsigned long long Ci = str2uint(temp);
        unsigned long long n_res = PowMod(Ci, private_key_d, n);
        unsigned short* p_res = (unsigned short*)&n_res;

        if (p_res[1] != 0 || p_res[2] != 0 || p_res[3] != 0)
        { // error
            // printf("sever server_decry() error!\n");
            return 0;
        }
        else // p_res[0]是16bit数字，可以转成2个字母
        {
            // cout << "before short2str" << endl;
            des_key += short2str(p_res[0]);
            // cout << "after short2str" << endl;
        }
    }
    // cout << "des_key" << des_key << endl;

    return des_key;
}