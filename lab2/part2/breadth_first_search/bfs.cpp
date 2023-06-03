#include "bfs.h"

#include <cstddef>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
// #define VERBOSE

void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, int *distances,
                   int iteration) {
  int local = 0;
#pragma omp parallel for reduction(+ : local)
  for (int i = 0; i < g->num_nodes; i++) {
    if (frontier->vertices[i] == iteration) {
      int start_edge = g->outgoing_starts[i];
      int end_edge =
          (i == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[i + 1];

      // attempt to add all neighbors to the new frontier
      for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int outgoing = g->outgoing_edges[neighbor];
        if (frontier->vertices[outgoing] == 0) {
          distances[outgoing] = distances[i] + 1;
          frontier->vertices[outgoing] = iteration + 1;
          local++;
        }

        // if (distances[outgoing] == NOT_VISITED_MARKER) {
        // #pragma omp critical
        //         {
        //           distances[outgoing] = distances[node] + 1;
        //           // int index = new_frontier->count++;
        //           // new_frontier->vertices[index] = outgoing;
        //         }
      }
    }
  }
  frontier->count = local;
}

void bottom_up_step(Graph g, vertex_set *frontier, int *distances,
                    int iteration) {
  int local = 0;
#pragma omp parallel for reduction(+ : local)
  for (int i = 0; i < num_nodes(g); i++) {
    if (frontier->vertices[i] == 0) {
      int stargEdge = g->incoming_starts[i];
      int endEdge =
          (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
      for (int neighbor = stargEdge; neighbor < endEdge; neighbor++) {
        int idex = g->incoming_edges[neighbor];
        if (frontier->vertices[idex] == iteration) {
          distances[i] = distances[idex] + 1;
          frontier->vertices[i] = iteration + 1;
          local++;
          break;
        }
      }
    }
  }
  frontier->count += local;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

  vertex_set list1;
  // vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  // vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  // vertex_set *new_frontier = &list2;
  int iteration = 1;
#pragma omp parallel for
  for (int i = 0; i < frontier->max_vertices; i++) {
    frontier->vertices[i] = 0;
  }

  frontier->vertices[frontier->count++] = iteration;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  // frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(frontier);

    top_down_step(graph, frontier, sol->distances, iteration);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    // vertex_set *tmp = frontier;
    // frontier = new_frontier;
    // new_frontier = tmp;
    iteration++;
  }
}

void bfs_bottom_up(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
  vertex_set list1;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set *frontier = &list1;
  int iteration = 1;
#pragma omp parallel for
  for (int i = 0; i < frontier->max_vertices; i++) {
    frontier->vertices[i] = 0;
  }
  frontier->vertices[frontier->count++] = iteration;
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;
  sol->distances[ROOT_NODE_ID] = 0;
  while (frontier->count != 0) {
    vertex_set_clear(frontier);
    bottom_up_step(graph, frontier, sol->distances, iteration);
    iteration++;
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  vertex_set list1;

  vertex_set_init(&list1, graph->num_nodes);

  vertex_set *frontier = &list1;
  if (frontier->max_vertices > 1e7)
    bfs_bottom_up(graph, sol);
  else
    bfs_top_down(graph, sol);
}
