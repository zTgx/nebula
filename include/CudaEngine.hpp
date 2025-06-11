class CudaWrapper {
public:
    CudaWrapper(int size);
    ~CudaWrapper();

    void copyDataToDevice(const int* hostData);
    void launchKernel(int numBlocks, int threadsPerBlock);
    void copyDataFromDevice(int* hostData);

private:
    int size_;
    int* deviceData_;
};