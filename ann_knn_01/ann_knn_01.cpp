#include <ANN/ANN.h>
#include <random>

int main(int argc, char** argv) {
    int k = 2;

    auto data = annAllocPts(4, 3);
    data[0][0] = 5.0;
    data[0][1] = 5.0;
    data[0][2] = 5.0;
    data[1][0] = 50.0;
    data[1][1] = 50.0;
    data[1][2] = 50.0;
    data[2][0] = 23.0;
    data[2][1] = 34.0;
    data[2][2] = 3.0;
    data[3][0] = 11.0;
    data[3][1] = 43.0;
    data[3][2] = 5.0;

    auto kd_tree = new ANNkd_tree(data, 4, 3);

    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_real_distribution<double> urd(0.0, 55.0);
    for(auto j = 0; j < 10; j++) {
        auto query = annAllocPt(3);
        query[0] = urd(gen);
        query[1] = urd(gen);
        query[2] = urd(gen);
        std::cout << std::endl << "(" << query[0] << ", " << query[1] << ", " << query[2] << ")" << std::endl;

        auto neighbors = new ANNidx[k];
        auto dists = new ANNdist[k];

        kd_tree->annkSearch(query, k, neighbors, dists);

        std::cout << "NN:\tIndex:\tDistance:" << std::endl;
        for(auto i = 0; i < k; i++) {
            dists[i] = sqrt(dists[i]);
            std::cout << i << "\t" << neighbors[i] << "\t" << dists[i] << std::endl;
        }

        delete[] dists;
        delete[] neighbors;
        delete[] query;
    }

    annClose();
}