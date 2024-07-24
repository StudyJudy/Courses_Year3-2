#include <iostream>
#include <bitset>
using namespace std;


typedef bitset<64> Block;// 64 bit����
typedef bitset<56> Key;
typedef bitset<48> Code;

typedef bitset<32> HBlock;// 32bit���ݿ�
typedef bitset<28> HKey;
typedef bitset<24> HCode;

typedef enum { e, d } Method;

int IP(const Block& block, HBlock& left, HBlock& right);
int Turn(HBlock& left, HBlock& right, const Code& subkey);
int exchange(HBlock& left, HBlock& right);
int Reverse_IP(const HBlock& left, const HBlock& right, Block& block);
Code getkey(const unsigned int n, const Block& bkey);
int des(Block& block, Block& bkey, const Method method);