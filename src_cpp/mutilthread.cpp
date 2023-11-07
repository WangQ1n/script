#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock
 
std::mutex mtx;           // mutex for critical section
 
void print_block (int n, char c) {
    // critical section (exclusive access to std::cout signaled by lifetime of lck):
    std::unique_lock<std::mutex> lck (mtx);
    for (int i=0; i<n; ++i) {
        std::cout << c;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << '\n';
}
 
void print_block_without_lock (int n, char c) {
    for (int i=0; i<n; ++i) {
        std::cout << c;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << '\n';
}

int main ()
{
    std::thread th1 (print_block,50,'*');
    std::thread th2 (print_block,50,'$');
 
    th1.join();
    th2.join();

    std::thread th3 (print_block_without_lock,50,'*');
    std::thread th4 (print_block_without_lock,50,'$');
 
    th3.join();
    th4.join();
 
    return 0;
}