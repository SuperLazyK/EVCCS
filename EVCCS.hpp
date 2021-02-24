#pragma once

#include <vector>
#include <set>
#include <list>
#include <string>
#include <limits>
#include <fstream>
#include <Eigen/Eigenvalues>
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/Geometry3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/Octree.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/utility/Console.h>
#include <open3d/io/PointCloudIO.h>
#include "json.hpp"
#include "glasbey_lut.hpp"

namespace open3d {

namespace geometry {

class SuperVoxel;

using nlohmann::json;

#define ASSERT(p) if(!(p)) {std::cout << __FILE__ ":" << __LINE__ << " " #p << std::endl; throw #p;}

static inline std::set<int> envvar(const char* name)
{
    const char* v = std::getenv(name);
    if (v) {
		std::string s(v);
		std::set<int> ret;
		std::stringstream ss(s);
		std::string item;
		while (getline(ss, item, ',')) {
			if (!item.empty()) {
				ret.insert(stoi(item));
			}
		}
		return ret;
	}
    return std::set<int>();
}

static inline int envvar(const char* name, int def)
{
    const char* v = std::getenv(name);
    if (v) {
        return std::stoi(v);
    }
    return def;
}

static inline double envvar(const char* name, double def)
{
    const char* v = std::getenv(name);
    if (v) {
        return std::stof(v);
    }
    return def;
}

static inline json vec2json(const Eigen::Vector3d v)
{
    return json::array({v(0), v(1), v(2)});
}


static inline Eigen::Vector3d rgb2xyz(const Eigen::Vector3d& rgb) {
    Eigen::Vector3d srgb;
    srgb(0)  = rgb(0) > 0.04045 ? std::pow((rgb(0) + 0.055) / 1.055, 2.4) : rgb(0) / 12.92;
    srgb(1)  = rgb(1) > 0.04045 ? std::pow((rgb(1) + 0.055) / 1.055, 2.4) : rgb(1) / 12.92;
    srgb(2)  = rgb(2) > 0.04045 ? std::pow((rgb(2) + 0.055) / 1.055, 2.4) : rgb(2) / 12.92;
    return Eigen::Vector3d(
        (srgb[0] * 0.4124) + (srgb[1] * 0.3576) + (srgb[2] * 0.1805),
        (srgb[0] * 0.2126) + (srgb[1] * 0.7152) + (srgb[2] * 0.0722),
        (srgb[0] * 0.0193) + (srgb[1] * 0.1192) + (srgb[2] * 0.9505)
    );
}

static inline Eigen::Vector3d xyz2lab(const Eigen::Vector3d& xyz) {
    Eigen::Vector3d xyzD;
    xyzD(0) = xyz(0) > 0.008856 ? std::cbrt(xyz(0)) : 7.787 * xyz(0) + 16 / 116;
    xyzD(1) = xyz(1) > 0.008856 ? std::cbrt(xyz(1)) : 7.787 * xyz(1) + 16 / 116;
    xyzD(2) = xyz(2) > 0.008856 ? std::cbrt(xyz(2)) : 7.787 * xyz(2) + 16 / 116;
    return Eigen::Vector3d((116.0 * xyzD[1]) - 16.0, 500.0 * (xyzD[0] - xyzD[1]), 200.0 * (xyzD[1] - xyzD[2]));
}

static inline Eigen::Vector3d rgb2lab(const Eigen::Vector3d& rgb) {
    return xyz2lab(rgb2xyz(rgb));
}


// average filter of region property
struct RegionProperty {
public:
    RegionProperty()
        : color_(Eigen::Vector3d::Zero())
        , point_(Eigen::Vector3d::Zero())
        , normal_(Eigen::Vector3d::Zero())
        , num_of_points_(0)
        {}

    RegionProperty(const Eigen::Vector3d& color, const Eigen::Vector3d& point, const Eigen::Vector3d& normal)
        : color_(color)
        , point_(point)
        , normal_(normal)
        , num_of_points_(1)
        {}

    inline RegionProperty operator+(const RegionProperty& rhs) const
    {
        RegionProperty ret;
        ret.color_  = color_  + rhs.color_ ;
        ret.point_  = point_  + rhs.point_ ;
        ret.normal_ = normal_ + rhs.normal_;
        ret.num_of_points_ = num_of_points_ + rhs.num_of_points_;
        return ret;
    }

    inline RegionProperty operator-(const RegionProperty& rhs) const
    {
        RegionProperty ret;
        ret.color_  = color_  - rhs.color_ ;
        ret.point_  = point_  - rhs.point_ ;
        ret.normal_ = normal_ - rhs.normal_;
        ret.num_of_points_ = num_of_points_ - rhs.num_of_points_;
        return ret;
    }

    inline RegionProperty Normalize()
    {
        RegionProperty ret;
        ASSERT(num_of_points_ > 0);
        ret.color_  = color_ / num_of_points_;
        ret.point_  = point_ / num_of_points_;
        ret.normal_ = normal_.normalized();
        ret.num_of_points_ = num_of_points_;
        return ret;
    }


    std::string toString() const
    {
        std::stringstream ss;
        ss << "p [" << point_(0) << ", " << point_(1) << ", " << point_(2) << "]";
        ss << "n [" << normal_(0) << ", " << normal_(1) << ", " << normal_(2) << "]";
        ss << "c [" << color_(0) << ", " << color_(1) << ", " << color_(2) << "]";
        ss << "n :" << num_of_points_;
        return ss.str();
    }

    inline double DistanceToPlane(const Eigen::Vector3d &p) const
    {
        return std::abs(normal_.dot(point_ - p));
    }

public:
    Eigen::Vector3d color_;
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
    int num_of_points_;

};


std::function<double(const RegionProperty&, const RegionProperty&)> DistanceFunctionL1(double lambda, double mu, double epsilon)
{
    std::function<double(const RegionProperty&, const RegionProperty&)> f
    = [lambda, mu, epsilon](const RegionProperty& r1, const RegionProperty& r2) -> double{
        double Dc = (r1.color_ - r2.color_).norm();
        double Ds = (r1.point_ - r2.point_).norm();
        double Dn = 1.0f - std::abs(r1.normal_.dot(r2.normal_));
        return  lambda * Dc + mu * Ds + epsilon * Dn;
    };
    return f;
}


std::function<double(const RegionProperty&, const RegionProperty&)> DistanceFunctionLInf(double lambda, double mu, double epsilon)
{
    std::function<double(const RegionProperty&, const RegionProperty&)> f
    = [lambda, mu, epsilon](const RegionProperty& r1, const RegionProperty& r2) -> double{
        double Dc = (r1.color_ - r2.color_).norm();
        double Ds = (r1.point_ - r2.point_).norm();
        double Dn = 1.0f - std::abs(r1.normal_.dot(r2.normal_));
        return  std::max(std::max(lambda * Dc, mu * Ds), epsilon * Dn);
    };
    return f;
}


template <typename T>
void BFS(T root, const std::vector<std::set<T> > &adj, int max_depth, const std::function<bool(const T&)> &f) {

    std::set<T> visited;
    std::list<T> leaves;

    leaves.push_back(root);
    int depth = 0;

    while (leaves.size() > 0) {

        if (depth == max_depth) {
            return;
        }

        std::list<T> new_leaves;

        for (auto leaf : leaves) {

            bool keep_searching = f(leaf);

            if (!keep_searching) {
                continue;
            }

            for (auto neighbor : adj[leaf]) {
                if (visited.find(neighbor) == visited.end()) {
                    new_leaves.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        leaves = new_leaves;
        depth++;
    }
}

static Eigen::Vector3d FlipDirection(const Eigen::Vector3d &target, const Eigen::Vector3d reference = Eigen::Vector3d(0, 0, 1), const Eigen::Vector3d second_reference = Eigen::Vector3d(0, 1, 0))
{
    if (target.dot(reference) < 0) {
        return -target;
    }
    if (target.dot(reference) > 0) {
        return target;
    }
    if (target.dot(second_reference) < 0) {
        return -target;
    }
    return target;
}



static Eigen::Vector3d EstimateNormal(
        const geometry::PointCloud& pc,
        const std::vector<int>& indices,
        Eigen::Vector3d &eigenvalues)
{
    if (indices.size() < 3) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Matrix3d covariance = utility::ComputeCovariance(pc.points_, indices);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.computeDirect(covariance, Eigen::ComputeEigenvectors);

    eigenvalues = solver.eigenvalues();

    return FlipDirection(solver.eigenvectors().col(0));
}

using GridKey = int64_t;
using PointID = int;
using VoxelID = int;
using SuperVoxelID = int;


struct SuperVoxel {
public:
    RegionProperty prop_;
    VoxelID seed_;
};


struct Edge {
public:
    Edge(SuperVoxelID svid1, SuperVoxelID svid2, double d) : svid1(svid1), svid2(svid2), d(d) {
        ASSERT(svid1 < svid2);
        ASSERT(d >= 0);
    }
    SuperVoxelID svid1;
    SuperVoxelID svid2;
    double d;
};

struct EdgeCompare {
public:
    inline bool operator()(const Edge& a, const Edge& b) const
    {
        return a.d < b.d;
    }
};


/// \class EVCCS
///
/// \brief Edge-removal Voxel Cloud Connectivity Segmentation.
///
class EVCCS
{
public:
    using LabelType = unsigned int;
    static const SuperVoxelID NO_LABEL = -1;
    static const VoxelID NO_VOXEL = -1;

public:
    /// \brief Default Constructor.
    EVCCS() { };

    ~EVCCS() { }

    // vxl_size : resolution.
    bool Execute(const PointCloud &pc, double vxl_size, int r_seed, int iteration_num, std::function<double(const RegionProperty&, const RegionProperty&)> f, double maxd, double r_adj)
    {
        utility::LogInfo("EVCCS: start");

        if ( !pc.HasColors() ) {
           utility::LogWarning("input point cloud does'nt have colors!");
           return false;
        }

        ASSERT(vxl_size > 0);
        ASSERT(r_seed > 1);

        int r_search = (r_seed+1) / 2;

        Voxelize(pc, vxl_size);

        ScatterSeeds(vxl_size, r_seed, r_search);

        FilterOutSeedsInSparseArea(vxl_size, r_search);

        //ShiftSeeds(r_search);

        Clustering(iteration_num, f, maxd, r_seed, r_adj);

        utility::LogInfo("EVCCS: end");
        return true;
    }

    //debug
public:
    //std::shared_ptr<PointCloud> CreateLabeledVoxelCloud(const std::vector<Eigen::Vector3d>& lut)
    //{
    //    Eigen::Vector3d nolabelColor(0, 0, 0);
    //    Eigen::Vector3d edgeColor(1, 1, 1);
    //    Eigen::Vector3d removalColor(1, 1, 1);
    //    auto pc = std::make_shared<PointCloud>();
    //    for(unsigned int i=0; i<vc_.points_.size(); i++) {
    //        if (static_cast<int>(i) == envvar("DEBUG_IDX", -1)) {
    //            std::cout << vc_.points_[i] << std::endl;
    //            if (vlabel_[i] == NO_LABEL) {
    //                std::cout << "nolabel" << std::endl;
    //            } else {
    //                std::cout << (lut[vlabel_[i] % lut.size()] * 255) << std::endl;
    //            }
    //        }
    //        pc->points_.push_back(vc_.points_[i]);
    //        pc->normals_.push_back(vc_.normals_[i]);
    //        if (vlabel_[i] == NO_LABEL) {
    //            pc->colors_.push_back(nolabelColor);
    //        } else {
    //            pc->colors_.push_back(lut[vlabel_[i] % lut.size()]);
    //        }
    //    }
    //    for(unsigned int i=0; i<evc_.points_.size(); i++) {
    //        pc->points_.push_back(evc_.points_[i]);
    //        pc->normals_.push_back(evc_.normals_[i]);
    //        pc->colors_.push_back(edgeColor);
    //    }
    //    for(unsigned int i=0; i<rvc_.points_.size(); i++) {
    //        pc->points_.push_back(rvc_.points_[i]);
    //        pc->normals_.push_back(rvc_.normals_[i]);
    //        pc->colors_.push_back(removalColor);
    //    }
    //    return pc;
    //}

    //void PaintLabelColor(PointCloud &pc, const std::vector<Eigen::Vector3d>& lut, double r = 0.02)
    //{
    //    Eigen::Vector3d nolabelColor(0, 0, 0);
    //    std::vector<int> vindices;
    //    std::vector<double> vdistance2;

    //    for(unsigned int i=0; i<pc.points_.size(); i++) {
    //        // TODO use normal and color in edge/removed-voxel kdtrree
    //        kdt_.SearchHybrid(pc.points_[i], r, 1, vindices, vdistance2);
    //        if (vindices.size() > 0) {
    //            ASSERT(vlabel_[vindices[0]] != NO_LABEL);
    //            pc.colors_[i] = lut[vlabel_[vindices[0]] % lut.size()];
    //        }
    //    }
    //}

    //void DumpLabel(const PointCloud &pc, std::string filename, double r_adj)
    //{
    //        std::ofstream lo(filename, std::ios_base::binary);
    //        std::vector<int> vindices;
    //        std::vector<double> vdistance2;

    //        for(unsigned int i=0; i<pc.points_.size(); i++) {
    //            int label = NO_LABEL;
    //            kdt_.SearchHybrid(pc.points_[i], r_adj, 1, vindices, vdistance2);
    //            if (vindices.size() > 0) {
    //                label = vlabel_[vindices[0]];
    //            }
    //            lo.write((const char*)&label, sizeof(label));
    //        }
    //}

    SuperVoxelID Lookup(SuperVoxelID label, const std::vector<SuperVoxelID> &ltree, const std::vector<double> &dtree)
    {
        SuperVoxelID prev = label;
        while (true) {
            label = ltree[prev];
            if (label == prev) {
                break;
            }
            prev = label;
        }
        return prev;
    }

    void DumpLabeledPLY(const PointCloud &pc, std::string filename, const std::vector<SuperVoxelID> &ltree, const std::vector<double> &dtree, int min_point_num)
    {
        utility::LogInfo("EVCCS: DumpLabeledPLY start");
        {
            std::ofstream o(filename, std::ios_base::binary);
            if (o.is_open()) {
                o << "ply" << std::endl;
                o << "format binary_little_endian 1.0" << std::endl;
                o << "element vertex " << pc.points_.size() << std::endl;
                o << "property float x" << std::endl;
                o << "property float y" << std::endl;
                o << "property float z" << std::endl;
                o << "property uchar red" << std::endl;
                o << "property uchar green" << std::endl;
                o << "property uchar blue" << std::endl;
                o << "property int32 label" << std::endl;
                o << "property uchar lred" << std::endl;
                o << "property uchar lgreen" << std::endl;
                o << "property uchar lblue" << std::endl;
                o << "end_header" << std::endl;
            } else {
                std::cout << "cannot open " << filename << std::endl;
                return;
            }
        }

		std::vector<SuperVoxelID> lut(svxls_.size());
		for (unsigned int i=0; i<lut.size(); i++) {
			lut[i] = Lookup(i, ltree, dtree);
			//std::cout << "lut[" << i << "] : " << lut[i] <<  " : #" <<mergedsvxls_[lut[i]].num_of_points_  << " d:" << dtree[lut[i]] << std::endl;
		}

		PointCloud big_vc;
		std::vector<SuperVoxelID> big_label;
		for (unsigned int i=0; i<vc_.points_.size(); i++ ) {
			auto label = lut[vlabel_[i]];
			//std::cout << "vxl[" << i << "] : " << vlabel_[i] << " -> " << label << " : " << mergedsvxls_[label].num_of_points_  << " big:" << (mergedsvxls_[label].num_of_points_ > min_point_num) << std::endl;
			if (mergedsvxls_[label].num_of_points_ > min_point_num) {
				big_vc.points_.push_back(vc_.points_[i]);
				big_label.push_back(label);
			}
		}
		KDTreeFlann big_kdt(big_vc);


        {
            std::set<int> targets = envvar("TARGET");
			double k = envvar("TARGET_RED", 0.5);
            std::ofstream o(filename, std::ios_base::binary|std::ios_base::app);
            std::vector<int> vindices;
            std::vector<double> vdistance2;

            for(unsigned int i=0; i<pc.points_.size(); i++) {
                int label = NO_LABEL;
                big_kdt.SearchKNN(pc.points_[i], 1, vindices, vdistance2);
				ASSERT(vindices.size() > 0);
				label = big_label[vindices[0]];
                for (int j=0; j<3; j++) {
                    float x = pc.points_[i](j);
                    o.write((const char*)&x, sizeof(x));
                }
                auto color = pc.colors_[i];
                if (targets.find(label) != targets.end()) {
                    //color(0) *= 0.5; 
                    color(1) *= k; 
                    color(2) *= k; 
                }
                for (int j=0; j<3; j++) {
                    unsigned char x = 255*color(j);
                    o.write((const char*)&x, sizeof(x));
                }
                o.write((const char*)&label, sizeof(label));
                auto labelcolor = GLASBEY_LUT[label % GLASBEY_LUT.size()];
                for (int j=0; j<3; j++) {
                    unsigned char x = 255 * labelcolor(j);
                    o.write((const char*)&x, sizeof(x));
                }
            }
        }
        utility::LogInfo("EVCCS: DumpLabeledPLY end");
    }


    void DumpSupervoxels(std::string filename)
    {
        json j;
        for (unsigned int svid=0; svid<svxls_.size(); svid++) {
            if (svxls_[svid]->seed_ == NO_VOXEL) {
                continue;
            }
            ASSERT(svxls_[svid]->prop_.num_of_points_ > 0);
            auto prop = svxls_[svid]->prop_.Normalize();
            j["node"][svid]["numOfPoints"] = prop.num_of_points_;
            j["node"][svid]["color"] = prop.color_;
            j["node"][svid]["point"] = prop.point_;
            j["node"][svid]["normal"] = prop.normal_;
        }

        std::ofstream o(filename);
        if (o.is_open()) {
            o << j;
        } else {
            std::cout << "cannot open " << filename << std::endl;
        }

    }

    void CreateLabelTree(
        std::vector<SuperVoxelID> &labeltree,
        std::vector<double> &disttree,
        std::function<double(const RegionProperty&, const RegionProperty&)> f,
		double  maxd
    )
    {
        utility::LogInfo("EVCCS: LabelTree start");

		mergedsvxls_.clear();
        labeltree.clear();
        disttree.clear();

        for (SuperVoxelID svid=0; svid<svxls_.size(); svid++) {
            if (svxls_[svid]->seed_ == NO_VOXEL) {
                mergedsvxls_.push_back(RegionProperty());
                labeltree.push_back(-1);
            } else {
                mergedsvxls_.push_back(svxls_[svid]->prop_);
                labeltree.push_back(svid);
            }
            disttree.push_back(0);
        }

        auto svadj = svadj_;
        std::multiset<Edge, EdgeCompare> edges;
        edges.clear();
        for (auto es : svadj) {
            SuperVoxelID svid1 = es.first;
            for (auto svid2 : es.second) {
                if (svid1 >= svid2) {
                    continue;
                }
                double d = f(mergedsvxls_[svid1], mergedsvxls_[svid2]);
                edges.insert(Edge(svid1, svid2, d));
                //std::cout << "init edge " << svid1 << " - " << svid2 << std::endl;
            }
        }

        while (edges.size() > 0) {

            auto e = edges.begin();
            edges.erase(e);
            auto svid1 = e->svid1;
            auto svid2 = e->svid2;
            ASSERT(svid1 <= labeltree[svid1]);
            ASSERT(svid2 <= labeltree[svid2]);
            //already merged
            if (svid1 != labeltree[svid1] || svid2 != labeltree[svid2]) {
                //std::cout << "ignore " << svid1 << "(->" <<  labeltree[svid1] << ") - " << svid2 << "(->" << labeltree[svid2] << ")" << std::endl;
                //std::cout << "ignoreDebug " << svid1 << " : " << labeltree[svid1] << " @ " << &labeltree[svid1] << std::endl;
                //std::cout << "ignoreDebug " << svid2 << " : " << labeltree[svid2] << " @ " << &labeltree[svid2] << std::endl;
                continue;
            }
			if (e->d > maxd) {
				break;
			}
            SuperVoxelID newID = mergedsvxls_.size();
            mergedsvxls_.push_back(mergedsvxls_[svid1] + mergedsvxls_[svid2]);

            if (envvar("DUMP_LABELTREE", 0)) {
                std::cout << "relabel " << svid1 << " (#" << mergedsvxls_[svid1].num_of_points_ << ") + " << svid2 << " (#" << mergedsvxls_[svid2].num_of_points_ << ") -> " << newID << " (#" << mergedsvxls_[newID].num_of_points_ << ") : " << e->d << std::endl;
            }

            labeltree[svid1] = newID;
            labeltree[svid2] = newID;
            labeltree.push_back(newID);
            //std::cout << "mergeDebug " << svid1 << " : " << labeltree[svid1] << " @ " << &labeltree[svid1] << std::endl;
            //std::cout << "mergeDebug " << svid2 << " : " << labeltree[svid2] << " @ " << &labeltree[svid2] << std::endl;
            //std::cout << "mergeDebug " << newID << " : " << labeltree[newID] << " @ " << &labeltree[newID] << std::endl;
            disttree.push_back(e->d);

            svadj[newID] = std::set<SuperVoxelID>();

            for (auto n1 : svadj[svid1] ) {
                if (n1 == svid2) { continue; }
                svadj[newID].insert(n1);
                svadj[n1].erase(svid1);
                svadj[n1].insert(newID);
                //std::cout << "adj : @" << svid1 << " " << n1 << "->" << newID << std::endl;
            }
            svadj.erase(svid1);

            for (auto n2 : svadj[svid2] ) {
                if (n2 == svid1) { continue; }
                svadj[newID].insert(n2);
                svadj[n2].erase(svid2);
                svadj[n2].insert(newID);
                //std::cout << "adj : @" << svid2 << " " << n2 << "->" << newID << std::endl;
            }
            svadj.erase(svid2);

            for (auto n : svadj[newID]) {
                double d = f(mergedsvxls_[newID].Normalize(), mergedsvxls_[n].Normalize());
                //std::cout << "edge : " << n << " " << newID << " " << d << std::endl;
                edges.insert(Edge(n, newID, d));
            }
            newID++;
        }
        utility::LogInfo("EVCCS: LabelTree end");
    }

protected:

    void Voxelize(const PointCloud &pc, double vxl_size, double size_expand = 0.01) {

        utility::LogInfo("voxelize");

        Eigen::Array3d min_bound = pc.GetMinBound();
        Eigen::Array3d max_bound = pc.GetMaxBound();
        Eigen::Array3d center = (min_bound + max_bound) / 2;
        Eigen::Array3d half_sizes = center - min_bound;
        double max_half_size = half_sizes.maxCoeff();
        Eigen::Vector3d origin = min_bound.min(center - max_half_size); // grid is isotropic box

        double size;
        if (max_half_size == 0) {
            size = size_expand;
        } else {
            size = max_half_size * 2 * (1 + size_expand);
        }

        double voxel_num = std::log2(std::ceil(size / vxl_size) - 1);
        const int MAX_VOXEL_NUM = std::numeric_limits<short>::max();
        if (voxel_num > MAX_VOXEL_NUM) {
            utility::LogWarning("vxl_size is too small.");
            while (voxel_num > MAX_VOXEL_NUM) {
                vxl_size *= 2;
                voxel_num = std::log2(std::ceil(size / vxl_size) - 1);
            }
            utility::LogWarning("vxl_size is clipped.");
        }

        int grid_size = 1 << (static_cast<int>(voxel_num) + 1);

        auto index = [origin, vxl_size](const Eigen::Vector3d& point) -> Eigen::Vector3i { return ((point - origin) / vxl_size).cast<int>(); };
        auto toKey = [](const Eigen::Vector3i& v) -> GridKey { return (static_cast<GridKey>(v(0)) << 32) + (static_cast<GridKey>(v(1)) << 16) + v(2); };

        std::map<GridKey, std::vector<PointID> > voxels;

        for (unsigned int i = 0; i < pc.points_.size(); i++) {
            auto point = pc.points_[i];
            GridKey key = toKey(index(point));
            voxels[key].push_back(i);
        }

        std::map<GridKey, VoxelID> idtbl;
        std::map<GridKey, VoxelID> idtbl_removed;

        utility::LogInfo("initialize voxel point cloud and index table");
        {
            vc_.Clear();
            //evc_.Clear();
            rvc_.Clear();

            for(auto p:voxels){
                GridKey key = p.first;
                auto indeces = p.second;
                ASSERT(indeces.size() > 0);
                Eigen::Vector3d eigenvalues(0, 0, 0);
                auto normal = EstimateNormal(pc, indeces, eigenvalues);

                Eigen::Vector3d color(0, 0, 0);
                Eigen::Vector3d point(0, 0, 0);
                for (auto i : indeces) {
                    color = color + pc.colors_[i];
                    point = point + pc.points_[i];
                }
                color = color / indeces.size();
                point = point / indeces.size();

                double var = 0;
                for (auto i : indeces) {
                    auto d = pc.colors_[i]-color;
                    var = var + d.dot(d);
                }
                var /= indeces.size();

                if (normal == Eigen::Vector3d::Zero() || eigenvalues(1) <= 0 ||  eigenvalues(2) <= 0) {
                    idtbl_removed[key] = rvc_.points_.size();
                    rvc_.points_.push_back(point);
                    rvc_.colors_.push_back(color);
                    continue;
                } else if (var >= envvar("THRES_COLOR_VAR", 0.05)) {
                    idtbl_removed[key] = rvc_.points_.size();
                    rvc_.points_.push_back(point);
                    rvc_.colors_.push_back(color);
                    continue;
                } else if (std::abs(eigenvalues(0) / eigenvalues(1)) >= envvar("THRES_LAMBDA", 0.01) || std::abs(eigenvalues(1) / eigenvalues(2)) < envvar("THRES_LAMBDA", 0.01)) {
                    idtbl_removed[key] = rvc_.points_.size();
                    rvc_.points_.push_back(point);
                    rvc_.colors_.push_back(color);
                    continue;
                } else if (eigenvalues(0) / (eigenvalues(0) + eigenvalues(1) + eigenvalues(2)) >= envvar("MAX_CURVATURE", 0.002)) {
                    idtbl_removed[key] = rvc_.points_.size();
                    rvc_.points_.push_back(point);
                    rvc_.colors_.push_back(color);
                    continue;
                } else {
                    idtbl[key] = vc_.points_.size();
                    vc_.points_.push_back(point);
                    vc_.colors_.push_back(color);
                    vc_.normals_.push_back(normal);
                    continue;
                }
            }
        }

        for (auto ir:idtbl_removed) {
            ASSERT(idtbl.find(ir.first) == idtbl.end());
            idtbl[ir.first]= ir.second + vc_.points_.size();
        }
        //std::cout << idtbl.size() << " " << vc_.points_.size() << " " << rvc_.points_.size() << std::endl;
        ASSERT(idtbl.size() == vc_.points_.size() + rvc_.points_.size());

        ASSERT(vc_.points_.size() <  std::numeric_limits<VoxelID>::max());
        ASSERT(vc_.points_.size() > 0);

        utility::LogInfo("construct kdtree");
        kdt_.SetGeometry(vc_);

        utility::LogInfo("constract voxel adjacency graph");
        rvadj_.clear();
        rvadj_.resize(idtbl.size());
        vadj_.clear();
        vadj_.resize(vc_.points_.size());
        for(unsigned int i=0; i< idtbl.size(); i++){
            Eigen::Vector3i idx;
            if (i < vc_.points_.size()) {
                idx = index(vc_.points_[i]);
            } else {
                idx = index(rvc_.points_[i - vc_.points_.size()]);
            }
            GridKey key = toKey(idx);
            ASSERT(idtbl[key] == static_cast<int>(i));
            for (int x=-1; x<=1; x++) {
                if (idx(0) + x < 0 || idx(0) + x >= grid_size) { continue; }
                for (int y=-1; y<=1; y++) {
                    if (idx(1) + y < 0 || idx(1) + y >= grid_size) { continue; }
                    for (int z=-1; z<=1; z++) {
                        if (idx(2) + z < 0 || idx(2) + z >= grid_size) { continue; }

                        Eigen::Vector3i neighboridx = idx + Eigen::Vector3i(x, y, z);
                        if (idx == neighboridx) { continue; }

                        GridKey nkey = toKey(neighboridx);

                        if (idtbl.find(nkey) != idtbl.end()) {
                            VoxelID nvxlid = idtbl[nkey];
                            if (nvxlid != static_cast<int>(i)) {
                                if (i < vc_.points_.size() && nvxlid < vc_.points_.size()) {
                                    vadj_[nvxlid].insert(i);
                                    vadj_[i].insert(nvxlid);
                                }
                                rvadj_[nvxlid].insert(i);
                                rvadj_[i].insert(nvxlid);
                            }
                        }
                    }
                }
            }
        }

        if (envvar("DUMP_ADJ", 0)) {
            for (unsigned int i=0; i<vadj_.size(); i++) {
                for (auto n : vadj_[i]) {
                    std::cout << "adj " << i << " -> " << n << std::endl;
                }
            }
        }

    }


    virtual void ScatterSeeds(double vxl_size, int r_seed, int r_search)
    {
        utility::LogInfo("scatter super voxel seeds");

        //auto center = vc_.GetCenter();
        auto center = (vc_.GetMaxBound() + vc_.GetMinBound()) / 2;
        auto extent = vc_.GetMaxBound() - vc_.GetMinBound();
        double r = vxl_size*r_seed;
        int xn2 = extent(0)/(2*r) + 1;
        int yn2 = extent(1)/(2*r) + 1;
        int zn2 = extent(2)/(2*r) + 1;

        std::set<VoxelID> seeds;
        for (int x=-xn2; x<=xn2; x++) {
            for (int y=-yn2; y<=yn2; y++) {
                for (int z=-zn2; z<=zn2; z++) {

                    Eigen::Vector3d point = center + Eigen::Vector3d(x*r, y*r, z*r);

                    std::vector<int> indices;
                    std::vector<double> distance2;

                    int k = kdt_.SearchHybrid(point, r, 1, indices, distance2);
                    if (k == 0) {
                        continue;
                    }
                    ASSERT(indices.size() == 1);
                    seeds.insert(indices[0]);
                }
            }
        }

        for (auto seed : seeds) {
            auto svxl = std::make_shared<SuperVoxel>();
            svxl->seed_ = seed;
            svxl->prop_ = MakeRegionProperty(seed);
            svxls_.push_back(svxl);
        }
    }

    RegionProperty MakeRegionProperty(VoxelID vid)
    {
        return RegionProperty(vc_.colors_[vid], vc_.points_[vid], vc_.normals_[vid]);
    }

    // NOTE: coeff == 0.1 in PCL implementation.
    virtual void FilterOutSeedsInSparseArea(double vxl_size, int r_search, float coeff = 0.2f)
    {
        utility::LogInfo("filter out super voxel seeds");
        float min_points = coeff * r_search * r_search * 3.1415926536f / 2;
        std::vector<std::shared_ptr<SuperVoxel> > new_svxls;

        for(auto svxl : svxls_) {
            std::vector<int> indices;
            std::vector<double> distance2;

            int k = kdt_.SearchRadius(svxl->prop_.point_, vxl_size * r_search, indices, distance2);

            if (k > min_points) {
                new_svxls.push_back(svxl);
                continue;
            }
        }
        svxls_ = new_svxls;
    }


    virtual void UpdateSeed()
    {
        for(auto svxl : svxls_) {
            if (svxl->seed_ == NO_VOXEL) {
                continue;
            }
            std::vector<int> indices;
            std::vector<double> distance2;
            kdt_.SearchKNN(svxl->prop_.Normalize().point_, 1, indices, distance2);
            if (vc_.points_[svxl->seed_] == svxl->prop_.point_) {
                ASSERT(svxl->seed_ == indices[0]);
            }
            svxl->seed_ = indices[0];
            svxl->prop_ = MakeRegionProperty(indices[0]);
        }
    }


    void Clustering(int iteration_num, std::function<double(const RegionProperty&, const RegionProperty&)> f, double max_d, int r_seed, double r_adj)
    {
        utility::LogInfo("clustering voxels into supervoxels");

        for (int iteration=0; iteration<iteration_num; iteration++) {
            UpdateSeed();

            std::vector<SuperVoxelID> v2l(vc_.points_.size(), NO_LABEL);; // voxelid -> supervoxel
            std::vector<double> v2d(vc_.points_.size(), std::numeric_limits<double>::infinity()); // voxelid -> minimum distance

            std::vector<std::vector<VoxelID> > sv2q(svxls_.size()); // search queues for each super voxel
            std::vector<RegionProperty> sv2p(svxls_.size()); // supervoxel -> new region property
            std::vector<std::set<VoxelID> > visited(svxls_.size());

            // initialize svxl propery, searchqueue, label table
            for (unsigned int i=0; i<svxls_.size(); i++) {
                SuperVoxelID svxlid = i;
                auto svxl = svxls_[svxlid];
                if (svxl->seed_ == NO_VOXEL) {
                    continue;
                }
                visited[svxlid].insert(svxl->seed_);
                sv2q[svxlid].push_back(svxl->seed_);
            }

            // kmean
            int depth = 0;
            while (true) {
                bool leaf_appended = false;
                // 1-layer bfs
                for (unsigned int i=0; i<svxls_.size(); i++) {
                    SuperVoxelID svxlid = i;

                    if (svxls_[svxlid]->seed_ == NO_VOXEL) {
                        continue;
                    }
                    if (svxls_[svxlid]->prop_.num_of_points_ == 0) {
                        svxls_[svxlid]->seed_ = NO_VOXEL;
                        continue;
                    }

                    auto cur_svxl_prop = svxls_[svxlid]->prop_.Normalize();

                    std::vector<VoxelID> newq;
                    auto leaves = sv2q[svxlid];

                    for (auto vid : leaves) {
                        auto prop = MakeRegionProperty(vid);
                        auto d = f(prop, cur_svxl_prop);
                        if (d > max_d) {
                            continue;
                        }
                        if (v2d[vid] > d) {
                            v2d[vid] = d;
                            if (v2l[vid] != NO_LABEL) {
                                ASSERT(svxlid != v2l[vid]);
                                auto old_svxlid = v2l[vid];
                                sv2p[old_svxlid] = sv2p[old_svxlid] - prop;
                            }
                            for (auto n : vadj_[vid]) {
                                if (visited[svxlid].find(n) == visited[svxlid].end()) {
                                    newq.push_back(n);
                                    leaf_appended = true;
                                    visited[svxlid].insert(n);
                                }
                            }
                            v2l[vid] = svxlid;
                            sv2p[svxlid] = sv2p[svxlid] + prop;
                        }
                    }

                    sv2q[svxlid] = newq;
                }

                //debug
                if (envvar("DUMP_CLUSTERING", 0)) {
                    std::cout << "---------------------" << std::endl;
                    std::cout << "iteration " << iteration << " depth " << depth << std::endl;
                    for (unsigned int i=0; i<svxls_.size(); i++) {
                        //if (i != 15) { continue; }
                        if (svxls_[i]->seed_ == NO_VOXEL) { continue; }
                        std::cout << "label  " << i  << std::endl;
                        std::cout << "seed " << svxls_[i]->prop_.toString() << std::endl;
                        std::cout << "prop " << sv2p[i].toString() << std::endl;
                        for (auto vid : visited[i]) {
                            if (v2l[vid] == static_cast<int>(i)) {
                                auto x = vc_.points_[vid](0);
                                auto y = vc_.points_[vid](1);
                                auto z = vc_.points_[vid](2);
                                auto nx = vc_.normals_[vid](0);
                                auto ny = vc_.normals_[vid](1);
                                auto nz = vc_.normals_[vid](2);
                                auto r = vc_.colors_[vid](0);
                                auto g = vc_.colors_[vid](1);
                                auto b = vc_.colors_[vid](2);
                                std::cout << "vxl " << vid  << " : [" <<  x << " " << y << " " << z << "] : [" <<  nx << " " << ny << " " << nz << "]" << "] : [" <<  r << " " << g << " " << b << "]" << std::endl;
                            }
                        }
                    }
                }

                depth++;

                if (!leaf_appended) {
                    break;
                }
            }

            // update label_ and property
            {
                vlabel_ = v2l;
                for (unsigned int i=0; i<svxls_.size(); i++) {
                    if (sv2p[i].num_of_points_ == 0) {
                        svxls_[i]->prop_ = RegionProperty();
                        svxls_[i]->seed_ = NO_VOXEL;
                    } else {
                        svxls_[i]->prop_ = sv2p[i];
                    }
                }
            }

            LabelAllVoxels(r_seed);
        }

        ComputeSuperVoxelAdjacency(r_adj);
    }


    void Label(VoxelID seed, SuperVoxelID svid, int r_seed)
    {
        auto f = [this, svid] (const VoxelID &v) -> bool {
            if (vlabel_[v] == NO_LABEL) {
                vlabel_[v] = svid;
                svxls_[svid]->prop_ = svxls_[svid]->prop_ + MakeRegionProperty(v);
                return true;
            }
            return false;
        };
        BFS<VoxelID>(seed, vadj_, r_seed, f);
    }


    void LabelAllVoxels(int r_seed)
    {
       for (unsigned int i=0; i<vlabel_.size(); i++) {
           if (vlabel_[i] == NO_LABEL) {
                SuperVoxelID freshID = svxls_.size();
                auto svxl = std::make_shared<SuperVoxel>();
                svxls_.push_back(svxl);
                Label(i, freshID, r_seed);
           }
       }
    }


    void ComputeSuperVoxelAdjacency(double r_adj)
    {
        auto vlabel = vlabel_;

        ASSERT( vlabel_.size() == vadj_.size());

        for(unsigned int i=0; i<rvc_.points_.size(); i++) {
            std::vector<int> indices;
            std::vector<double> distance2;
            int k = kdt_.SearchKNN(rvc_.points_[i], 1, indices, distance2);
            vlabel.push_back(vlabel[indices[0]]);
        }

        for (unsigned int vid=0; vid<rvadj_.size(); vid++) {
            SuperVoxelID svid = vlabel[vid];
            if (svid == NO_LABEL) {
                continue;
            }
            ASSERT(svxls_[svid]->seed_ != NO_VOXEL);
            for (auto nvid :rvadj_[vid]) {
                SuperVoxelID nsvid = vlabel[nvid];
                if (nsvid == NO_LABEL) {
                    continue;
                }
                ASSERT(svxls_[nsvid]->seed_ != NO_VOXEL);
                //std::cout << vid << "[" << svid << "] -> " << nvid << "[" << nsvid << "]" << std::endl;
                if (svid != nsvid) {
                    //std::cout << svid << " -> " << nsvid  << std::endl;
                    svadj_[svid].insert(nsvid);
                }
            }
        }

    }



public:
    //super voxel
    KDTreeFlann kdt_;
    PointCloud vc_; // voxel cloud
    PointCloud rvc_; // debug : removed voxel cloud
    std::vector<std::set<VoxelID> > vadj_; // voxel -> neighbor voxels(only stable normal voxel)
    std::vector<std::set<VoxelID> > rvadj_; // voxel -> neighbor voxels
    std::vector<std::shared_ptr<SuperVoxel> > svxls_;
    std::vector<SuperVoxelID> vlabel_; // voxelid -> supervoxel
    std::map<SuperVoxelID, std::set<SuperVoxelID> > svadj_; // supervoxel -> neighbor supervoxels

	std::vector<RegionProperty> mergedsvxls_;
};

}  // namespace geometry
}  // namespace open3d
