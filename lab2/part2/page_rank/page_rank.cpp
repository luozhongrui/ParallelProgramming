#include "page_rank.h"

#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  int numNodes = num_nodes(g);
  double *score = new double[numNodes];
  bool converged = false;
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  
  while (!converged) {
    double globalDiff = 0.0;
    double newScore = 0.0;
    #pragma omp parallel for
    for(int i = 0; i < numNodes; i++){
      score[i] = solution[i];
    }
    // #pragma omp parallel for reduction(+: newScore)
    // for(int i = 0; i < numNodes; i++){
    //   if(outgoing_size(g, i) == 0){
    //     newScore += damping * score[i] / numNodes;
    //   }
    // }
    #pragma omp parallel for 
    for(int i = 0; i < numNodes; i++){
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      double sum = 0.0;
      for(auto j = start; j != end;j++){
        sum += score[*j] / outgoing_size(g, *j);
      }
      sum = (damping * sum) + (1.0 - damping) / numNodes;
      // sum += newScore;
      solution[i] = sum;
      // globalDiff += fabs(sum - score[i]);
    }

    #pragma omp parallel for reduction(+: newScore)
    for(int i = 0; i < numNodes; i++){
      if(outgoing_size(g, i) == 0){
        newScore += damping * score[i] / numNodes;
      }
    }
    #pragma omp parallel for reduction(+: globalDiff)
    for(int i = 0; i < numNodes; i++){
      solution[i] += newScore;
      globalDiff += fabs(solution[i] - score[i]);
    }

    converged = globalDiff < convergence;
  }
  delete[] score;

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi])
     }; converged = (global_diff < convergence)
     }

   */
}
