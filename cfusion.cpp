#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nifty/nifty/graph/opt/multicut/fusion_move.hxx>
#include <nifty/nifty/graph/opt/multicut/multicut_base.hxx>
#include <nifty/nifty/graph/opt/multicut/kernighan_lin.hxx>
#include <nifty/nifty/graph/opt/multicut/multicut_greedy_additive.hxx>
#include <nifty/graph/opt/common/solver_factory.hxx>

namespace py = pybind11;
namespace mc = nifty::graph::opt::multicut;

using GraphType = nifty::graph::UndirectedGraph<>;
using MulticutObjective = mc::MulticutObjective<GraphType, double>;
using int_type = py::ssize_t;
using float_type = float;

using int_arr_type = py::array_t<int_type, py::array::c_style | py::array::forcecast>;
using float_arr_type = py::array_t<float_type, py::array::c_style | py::array::forcecast>;

py::array_t<int_type> fuse(int_type num_nodes, const int_arr_type &edges, const float_arr_type &weights, const int_arr_type &proposals) {
  if (num_nodes < 1) {
    throw std::invalid_argument("number of nodes should be 1 or greater");
  }
  if (edges.ndim() != 2) {
    throw std::invalid_argument("edges should be a 2-dimensional array");
  }
  if (edges.shape(1) != 2) {
    throw std::invalid_argument("edges should have shape (number_of_edges, 2)");
  }
  if (weights.ndim() != 1) {
    throw std::invalid_argument("weights should be a 1-dimensional array");
  }
  if (edges.shape(0) != weights.shape(0)) {
    throw std::invalid_argument("number of edges in 'edges' and 'weights' do not match");
  }
  if (proposals.ndim() != 2) {
    throw std::invalid_argument("proposals should be a 2-dimensional array");
  }
  if (proposals.shape(0) < 2) {
    throw std::invalid_argument("at least 2 proposals are required");
  }
  if (proposals.shape(1) != num_nodes) {
    throw std::invalid_argument("number of nodes in each proposal should be equal to the number of nodes in the graph");
  }

  auto num_edges = edges.shape(0);
  GraphType graph(num_nodes, num_edges / 2);

  {
    auto e = edges.unchecked<2>();
    for (decltype(num_edges) i = 0; i < num_edges; ++i) {
      graph.insertEdge(e(i, 0), e(i, 1));
    }
  }

  MulticutObjective objective(graph);
  auto &obj_w = objective.weights();

  std::cerr << "obj_w.size() = " << obj_w.size() << std::endl;

  {
    auto w = weights.unchecked<1>();
    std::cerr << "w.shape(0) = " << w.shape(0) << std::endl;
    std::cerr << "num_edges = " << num_edges << std::endl;
    std::cerr << "weights begin" << std::endl;
    for (decltype(num_edges) i = 0; i < num_edges / 2; ++i) {
      obj_w[i] = w(i);
      std::cerr << i << ": " << obj_w[i] << std::endl;
    }
    std::cerr << "weights end" << std::endl;
  }


  using solver_type = mc::KernighanLin<MulticutObjective>;
  solver_type::SettingsType settings;
  settings.verbose = true;
  settings.epsilon = 1e-5;

  using solver_factory_type = nifty::graph::opt::common::SolverFactory<solver_type>;

  using fusion_move_type = mc::FusionMove<MulticutObjective>;

  fusion_move_type::SettingsType fusion_settings;
  fusion_settings.mcFactory = std::make_shared<solver_factory_type>(settings);

  using labels_type = fusion_move_type::NodeLabelsType;
  labels_type result(num_nodes);

  auto num_proposals = proposals.shape(0);

  std::vector<labels_type> proposals_in(num_proposals);
  std::vector<const labels_type *> proposals_ptrs(num_proposals);

  {
    auto p = proposals.unchecked<2>();
    for (std::size_t i = 0; i < num_proposals; ++i) {
      proposals_in[i].resize(num_nodes);
      for (std::size_t j = 0; j < num_nodes; ++j) {
        proposals_in[i][j] = p(i, j);
      }
      proposals_ptrs[i] = &proposals_in[i];
    }
  }

  std::cerr << "input preparation done" << std::endl;

  mc::FusionMove<MulticutObjective> fusion(objective, fusion_settings);
  fusion.fuse(proposals_ptrs, &result);

  std::cerr << "fusion done" << std::endl;

  py::array_t<int_type> output(num_nodes);

  {
    auto o = output.mutable_unchecked<1>();
    for (std::size_t i = 0; i < num_nodes; ++i) {
      o(i) = result[i];
    }
  }

  std::cerr << "result conversion done" << std::endl;

  return output;
}

PYBIND11_MODULE(cfusion, m) {
  m.doc() = "Module docstring.";
  m.def("fuse", &fuse, "Run docstring.");
}
