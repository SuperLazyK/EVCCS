#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <random>
#include <gtest/gtest.h>

#include "EVCCS.hpp"

namespace open3d {
namespace tests {

enum {
    DIR_Z = 2,
};

void fill_rand(std::vector<double>& xs, const double minv, const double maxv, int seed=0)
{
    std::mt19937 mt;
    mt.seed(seed);
    double scale = (maxv - minv)/0xffffffff;
    for (double& x : xs) {
        x = scale * mt() + minv;
    }
}

void fill_rand(std::vector<Eigen::Vector3d>& xs, const Eigen::Vector3d minv, const Eigen::Vector3d maxv, int seed=0)
{
    std::mt19937 mt;
    mt.seed(seed);
    Eigen::Vector3d scale = (maxv - minv)/0xffffffff;
    for (Eigen::Vector3d& x : xs) {
        double px = scale(0) * mt() + minv(0);
        double py = scale(1) * mt() + minv(1);
        double pz = scale(2) * mt() + minv(2);
        x = Eigen::Vector3d(px, py, pz);
    }
}

static inline void draw_sphere(geometry::PointCloud &pc, int n, double r, const Eigen::Vector3d center, const Eigen::Vector3d color) {
    std::vector<double> theta1;
    std::vector<double> theta2;
    theta1.resize(n);
    theta2.resize(n);
    fill_rand(theta1, 0, 2*M_PI, 0);
    fill_rand(theta2, 0, 2*M_PI, 1);
    for (int i=0; i<n; i++) {
        auto t1 = theta1[i];
        auto t2 = theta2[i];
        Eigen::Vector3d point = center + Eigen::Vector3d(sin(t2)*cos(t1)*r, sin(t2)*sin(t1)*r, cos(t2)*r);
        pc.points_.push_back(point);
        pc.colors_.push_back(color);
    }
}

static inline void draw_plane(geometry::PointCloud &pc, int n, double r, const Eigen::Vector3d center, int dir, const Eigen::Vector3d color, int seed=0) {
    Eigen::Vector3d minv(-r, -r, -r);
    Eigen::Vector3d maxv(r, r, r);
    minv(dir) = 0;
    maxv(dir) = 0;
    minv = minv + center;
    maxv = maxv + center;

    std::vector<Eigen::Vector3d> points;
    points.resize(n);
    fill_rand(points, minv, maxv, seed);

    for (int i=0; i<n; i++) {
        pc.points_.push_back(points[i]);
        pc.colors_.push_back(color);
    }
}


static inline void draw_corner(geometry::PointCloud &pc, int n, double r, const Eigen::Vector3d center) {

    std::vector<Eigen::Vector3d> points;
    points.resize(n);

    for (int dir=0; dir<3; dir++) {
        Eigen::Vector3d offset(r, r, r);
        offset(dir) = 0;
        Eigen::Vector3d color(1, dir%2, dir/2);
        draw_plane(pc, n, r, center + offset, dir, color, dir);
    }
}


static inline void draw_4planes(geometry::PointCloud &pc, int n, double r, const Eigen::Vector3d center) {
    for (int i=0; i<4; i++) {
        Eigen::Vector3d offset(r, r, 0);
        Eigen::Vector3d color(1, 0, 0);
        if (i%2 == 0) {
            offset(0) = -r;
            color(1) = 1;
        }
        if (i/2 == 0) {
            offset(1) = -r;
            color(2) = 1;
        }
        draw_plane(pc, n, r, offset + center, DIR_Z, color, i);
    }
}


static inline void draw_box(geometry::PointCloud &pc, int n, double r, const Eigen::Vector3d center, const Eigen::Vector3d color, int contact_area_idx = -1) {

    std::vector<Eigen::Vector3d> points;
    points.resize(n);

    int idx =0;
    for (int i=0; i<2; i++) {
        for (int dir=0; dir<3; dir++, idx++) {
            if (contact_area_idx == idx) {
                continue;
            }
            Eigen::Vector3d offset(0, 0, 0);
            offset(dir) = i == 0 ? r : -r;
            draw_plane(pc, n, r, center + offset, dir, color, dir);
        }
    }
}


static inline void draw_cube_on_plane(geometry::PointCloud &pc) {
    int n = 2000;
    double r = 0.2;
    draw_plane(pc, 4*n, 0.4, Eigen::Vector3d(0, 0, 0), DIR_Z, Eigen::Vector3d(0, 0, 0), 0);
    draw_box(pc, n, r, Eigen::Vector3d(0, 0, r), Eigen::Vector3d(0, 0, 0), 5);
}

static void executeEVCCS(geometry::PointCloud &pc, std::string testname, bool dumpInput=true)
{
    double voxel_size = open3d::geometry::envvar("VXL_SIZE", 0.03);
    int r_seed = open3d::geometry::envvar("R_SEED", 10);
    int iternum = open3d::geometry::envvar("ITERNUM", 3);
    double r_adj = r_seed * voxel_size;

    bool linf = open3d::geometry::envvar("DIST_LINF", 1);
    double lambda = open3d::geometry::envvar("DIST_LAMBDA", 5.0); // less than 30 * sqrt(3) rgb euclid
    double mu = open3d::geometry::envvar("DIST_MU", 1 / (r_seed * 2 * voxel_size)); // less than 2voxel
    double epsilon = open3d::geometry::envvar("DIST_EPSILON", 50.0); // less than 12degree
    double maxd = open3d::geometry::envvar("MAX_DIST", 1.0);

    utility::LogInfo("EVCCS: distance: linf=%d lambda=%f mu=%f epsilon=%f maxd=%f", linf, lambda, mu, epsilon, maxd);
    auto f = linf ? open3d::geometry::DistanceFunctionLInf(lambda, mu, epsilon) : open3d::geometry::DistanceFunctionL1(lambda, mu, epsilon);

    geometry::EVCCS vccs;
    vccs.Execute(pc, voxel_size, r_seed, iternum, f, maxd, r_adj);

    if (dumpInput) { io::WritePointCloud(testname + "Input.pcd", pc); }

    //auto lvc = vccs.CreateLabeledVoxelCloud(GLASBEY_LUT);
    //io::WritePointCloud(testname + "LabelVoxel.pcd", *lvc);

    //vccs.DumpSupervoxels(testname + ".json");
    //vccs.DumpLabel(pc, testname + ".label", r_adj);
    //vccs.PaintLabelColor(pc, GLASBEY_LUT, 3*voxel_size);
    //io::WritePointCloud(testname + "Label.pcd", pc);

    std::vector<int> labeltree;
    std::vector<double> disttree;

    double d = open3d::geometry::envvar("MERGE_THRES", 2.0);

    vccs.CreateLabelTree(labeltree, disttree, open3d::geometry::DistanceFunctionLInf(lambda, 0, epsilon), d);

    std::ofstream ltree(testname + ".ltree", std::ios_base::binary);
    std::ofstream dtree(testname + ".dtree", std::ios_base::binary);
    ltree.write((const char*)&labeltree[0], labeltree.size() * sizeof(labeltree[0]));
    dtree.write((const char*)&disttree[0], disttree.size() * sizeof(disttree[0]));

    unsigned int N = open3d::geometry::envvar("MERGE_MIN_POINTS", 100);
    vccs.DumpLabeledPLY(pc, testname + ".ply", labeltree, disttree, N);
}

TEST(EVCCS, Sphere) {
    geometry::PointCloud pc;
    draw_sphere(pc, 16000, 0.2, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0));
    draw_sphere(pc, 4000, 0.1, Eigen::Vector3d(0.9, 0.9, 0.9), Eigen::Vector3d(0.0, 1.0, 0.0));
    executeEVCCS(pc, "Sphere"); // why is test_info_->test_suite_name() NULL??
}

TEST(EVCCS, Corner) {
    geometry::PointCloud pc;
    draw_corner(pc, 2000, 0.2, Eigen::Vector3d(0.0, 0.0, 0.0));
    executeEVCCS(pc, "Corner");
}

TEST(EVCCS, Tile) {
    geometry::PointCloud pc;
    draw_4planes(pc, 2000, 0.2, Eigen::Vector3d(0.0, 0.0, 0.0));
    executeEVCCS(pc, "Tile");
}

TEST(EVCCS, Desk) {
    geometry::PointCloud pc;
    draw_cube_on_plane(pc);
    executeEVCCS(pc, "Desk");
}

TEST(EVCCS, Room) {
    geometry::PointCloud pc;
    io::ReadPointCloud("../data/room.pcd", pc);
    executeEVCCS(pc, "Room");
}

TEST(EVCCS, JSON) {
    using nlohmann::json;
    json j;
    j["edge"]["a"] = 3;
    j["edge"]["b"][1] = 3;
    j["edge"]["c"] = geometry::vec2json(Eigen::Vector3d(0, 1, 2));
    std::cout << j;
}

TEST(EVCCS, Clustering) {
    using namespace geometry;
    EVCCS vccs;
    for (int i=0; i<6; i++) {
        vccs.svxls_.push_back(std::make_shared<SuperVoxel>());
        vccs.svadj_[i] = std::set<int>();
    }
    vccs.svxls_[0]->prop_ = RegionProperty(Eigen::Vector3d(0, 0, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));
    vccs.svxls_[1]->prop_ = RegionProperty(Eigen::Vector3d(3, 0, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));
    vccs.svxls_[2]->prop_ = RegionProperty(Eigen::Vector3d(1, 1, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));
    vccs.svxls_[3]->prop_ = RegionProperty(Eigen::Vector3d(0, 4, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));
    vccs.svxls_[4]->prop_ = RegionProperty(Eigen::Vector3d(4, 4, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));
    vccs.svxls_[5]->prop_ = RegionProperty(Eigen::Vector3d(5, 3, 0) / 10, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));

    vccs.svadj_[0].insert(2);
    vccs.svadj_[1].insert(2);
    vccs.svadj_[2].insert(0);
    vccs.svadj_[2].insert(1);
    vccs.svadj_[2].insert(3);
    vccs.svadj_[3].insert(2);
    vccs.svadj_[3].insert(4);
    vccs.svadj_[3].insert(5);
    vccs.svadj_[4].insert(3);
    vccs.svadj_[4].insert(5);
    vccs.svadj_[5].insert(3);
    vccs.svadj_[5].insert(4);
    std::vector<int> labeltree;
    std::vector<double> disttree;
    vccs.CreateLabelTree(labeltree, disttree, open3d::geometry::DistanceFunctionLInf(5, 0, 50), 2.0);
    for (unsigned int i=0; i<labeltree.size(); i++) {
        std::cout << i << "->" << labeltree[i] << " @ " << disttree[i] << std::endl;
    }
}

}  // namespace tests
}  // namespace open3d

