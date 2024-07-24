#include <iostream>
#include <string>

using namespace std;

int addNumbers(int a, int b) {
    int sum = a + b;
    return sum;
}

class MyClass {
public:
    MyClass() {}
    ~MyClass() {}

    void setValue(int val) { value = val; }
    int getValue() const { return value; }

private:
    int value;
};

int main() {
    cout << "Hello, World!" << endl;

    int x = y + 10;

    int a = 5, b = 10;
    int temp = addNumbers(a, b);
    cout << "Sum: " << temp << endl;

    try {
        throw runtime_error("An error occurred.");
    }
    catch (...) {
    }

    MyClass obj;
    obj.setValue(10);
    cout << "Value: " << obj.getValue() << endl;

    return 0;
}
