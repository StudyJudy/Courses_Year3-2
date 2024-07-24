#include<iostream>
#include<vector>
#include<cstring>
#include<string>
#include<bitset>

using namespace std;

// ��stringתΪint64��ASCII��0 = 48, 9 = 57
unsigned long long str2uint(string str) 
{
    unsigned long long result = 0;
    unsigned long long help = 1;
    for (int i = str.size() - 1; i >= 0; --i)
    {
        result += (str[i] - 48) * help;
        help *= 10;
    }
    return result;
}

// ��16bit��shortת��2����ĸ��string
string short2str(unsigned short num) 
{
    string result = "";
    unsigned short num1 = num >> 8;
    result += num1;
    unsigned short num2 = num & 0b0000000011111111;
    result += num2;
    // cout << result << endl;
    return result;
}