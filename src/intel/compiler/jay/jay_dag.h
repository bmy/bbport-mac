/*
 * Copyright 2026 Intel Corporation
 * Copyright 2019 Broadcom
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/u_dynarray.h"

struct jay_edge {
   uint32_t node:31;
   bool strong  :1;
};

struct jay_dag {
   struct util_dynarray edges;
   uint32_t *adjacency;
   uint32_t node, node_count;
};

struct jay_dag_iterator {
   const struct jay_dag *dag;
   struct util_dynarray heads;
   uint32_t *parent_counts;
};

static inline void
jay_dag_init(struct jay_dag *dag, void *memctx, uint32_t node_count)
{
   assert(node_count >= 1 && "node 0 is reserved and always present");

   *dag = (struct jay_dag) {
      .adjacency = rzalloc_array(memctx, uint32_t, node_count),
      .node_count = node_count,
      .node = 1,
   };

   util_dynarray_init(&dag->edges, memctx);
}

static inline void
jay_dag_transpose(struct jay_dag *out, const struct jay_dag *in)
{
   jay_dag_init(out, in->edges.mem_ctx, in->node_count);

   /* Initialize the output edge array so we can do random access writes */
   memset(util_dynarray_grow_bytes(&out->edges, in->edges.size, 1), 0,
          in->edges.size);

   /* Determine the number of edges for each node after transpose */
   uint32_t *count = calloc(in->node_count, sizeof(uint32_t));
   util_dynarray_foreach(&in->edges, struct jay_edge, edge) {
      count[edge->node]++;
   }

   /* Prefix sum to get the layout of adjacency[] */
   unsigned n = 0;
   for (unsigned i = 0; i < in->node_count; ++i) {
      n += count[i];
      out->adjacency[i] = n;
   }

   /* Now copy the edges into place */
   for (uint32_t i = 0; i < in->node_count; ++i) {
      uint32_t first_adj = i > 0 ? in->adjacency[i - 1] : 0;

      for (unsigned j = first_adj; j < in->adjacency[i]; ++j) {
         struct jay_edge *node =
            util_dynarray_element(&in->edges, struct jay_edge, j);
         assert(node && count[node->node] > 0 && "exact calculations");
         count[node->node]--;
         uint32_t idx = out->adjacency[node->node - 1] + count[node->node];
         *util_dynarray_element(&out->edges, struct jay_edge, idx) =
            (struct jay_edge) {
               .node = i,
               .strong = node->strong,
            };
      }
   }
}

static inline void
jay_dag_iterator_init(struct jay_dag_iterator *it, const struct jay_dag *dag)
{
   *it = (struct jay_dag_iterator) {
      .dag = dag,
      .parent_counts =
         rzalloc_array(dag->edges.mem_ctx, uint32_t, dag->node_count),
   };

   util_dynarray_init(&it->heads, dag->edges.mem_ctx);
}

static inline void
jay_dag_add_edge(struct jay_dag *dag, uint32_t child, bool strong)
{
   if (child && child != dag->node) {
      assert(child < dag->node_count);

      /* We have to prune degenerate or duplicate edges */
      for (uint32_t i = (dag->node > 0 ? dag->adjacency[dag->node - 1] : 0);
           i < util_dynarray_num_elements(&dag->edges, struct jay_edge); ++i) {
         struct jay_edge *other =
            util_dynarray_element(&dag->edges, struct jay_edge, i);
         if (other->node == child) {
            other->strong |= strong;
            return;
         }
      }

      struct jay_edge edge = { .node = child, .strong = strong };
      util_dynarray_append(&dag->edges, edge);
   }
}

static inline void
jay_dag_next_node(struct jay_dag *dag)
{
   assert(dag->node < dag->node_count);

   dag->adjacency[dag->node++] =
      util_dynarray_num_elements(&dag->edges, struct jay_edge);
}

static inline void
jay_dag_iterate(struct jay_dag_iterator *it, uint32_t first, uint32_t last)
{
   assert(it->heads.size == 0 && "must be zeroed on entry");
   uint32_t first_adj = first > 0 ? it->dag->adjacency[first - 1] : 0;

   for (unsigned i = first_adj; i < it->dag->adjacency[last]; ++i) {
      struct jay_edge *edge =
         util_dynarray_element(&it->dag->edges, struct jay_edge, i);
      it->parent_counts[edge->node]++;
   }

   for (uint32_t i = last; i >= first; --i) {
      if (it->parent_counts[i] == 0) {
         util_dynarray_append(&it->heads, i);
      }
   }
}

/**
 * Removes a DAG head and moves any new dag heads into the heads list.
 */
static inline void
jay_dag_take_head(struct jay_dag_iterator *it, uint32_t head)
{
   assert(!it->parent_counts[head]);
   util_dynarray_delete_unordered(&it->heads, uint32_t, head);
   uint32_t first = head > 0 ? it->dag->adjacency[head - 1] : 0;

   for (unsigned i = first; i < it->dag->adjacency[head]; ++i) {
      uint32_t node =
         util_dynarray_element(&it->dag->edges, struct jay_edge, i)->node;

      if ((--it->parent_counts[node]) == 0) {
         util_dynarray_append(&it->heads, node);
      }
   }
}

static inline void
jay_dag_iterator_reset(struct jay_dag_iterator *it)
{
   while (it->heads.size) {
      jay_dag_take_head(it, util_dynarray_top(&it->heads, uint32_t));
   }
}

static inline void
jay_dag_print(struct jay_dag *dag)
{
   for (unsigned i = 0; i < dag->node_count; ++i) {
      uint32_t first = i > 0 ? dag->adjacency[i - 1] : 0;
      for (unsigned j = first; j < dag->adjacency[i]; ++j) {
         struct jay_edge *it =
            util_dynarray_element(&dag->edges, struct jay_edge, j);
         printf("%u->%u%s\n", i, it->node, it->strong ? "" : " (weak)");
      }
   }
}

#define jay_dag_foreach_edge(dag, head, it)                                    \
   for (struct jay_edge *it = ((struct jay_edge *) (dag)->edges.data) +        \
                              ((head) > 0 ? (dag)->adjacency[(head) - 1] : 0); \
        it <                                                                   \
        ((struct jay_edge *) (dag)->edges.data) + ((dag)->adjacency[(head)]);  \
        ++it)
