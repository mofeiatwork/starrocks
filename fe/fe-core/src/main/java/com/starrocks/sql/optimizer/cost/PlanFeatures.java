// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.starrocks.sql.optimizer.cost;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.starrocks.common.TreeNode;
import com.starrocks.sql.optimizer.ExpressionContext;
import com.starrocks.sql.optimizer.OptExpression;
import com.starrocks.sql.optimizer.OptExpressionVisitor;
import com.starrocks.sql.optimizer.operator.OperatorType;
import com.starrocks.sql.optimizer.statistics.Statistics;

import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Features for physical plan
 */
public class PlanFeatures {

    // A trivial implement of feature extracting
    // TODO: implement sophisticated feature extraction methods
    public static FeatureVector flattenFeatures(OptExpression plan) {
        Extractor extractor = new Extractor();
        PlanTreeBuilder builder = new PlanTreeBuilder();
        OperatorWithFeatures root = plan.getOp().accept(extractor, plan, builder);

        // summarize by operator type
        Map<OperatorType, List<Integer>> sumVector = Maps.newHashMap();
        sumByOperatorType(root, sumVector);
        int dummyLength = OperatorFeatures.numFeatures();
        // Generate a list of integers of length dummyLength filled with 0
        List<Integer> dummyList = IntStream.range(0, dummyLength).map(i -> 0).boxed().toList();

        // transform into a equal-size vector
        List<Integer> result = Lists.newArrayList();
        for (int start = OperatorType.PHYSICAL.ordinal();
                start < OperatorType.SCALAR.ordinal();
                start++) {
            result.add(start);
            List<Integer> vector = sumVector.get(start);
            if (vector != null) {
                result.addAll(vector);
            } else {
                result.addAll(dummyList);
            }
        }

        return new FeatureVector(result);
    }

    private static void sumByOperatorType(OperatorWithFeatures tree,
                                          Map<OperatorType, List<Integer>> sum) {
        List<Integer> vector = tree.toVector();
        OperatorType opType = tree.features.opType;
        List<Integer> exist = sum.computeIfAbsent(opType, (x) -> Lists.newArrayList());
        for (int i = 0; i < exist.size(); i++) {
            exist.set(i, vector.get(i) + exist.get(i));
        }

        // recursive
        for (var child : tree.getChildren()) {
            sumByOperatorType(child, sum);
        }
    }

    public static class FeatureVector {
        List<Integer> vector;

        public FeatureVector(List<Integer> vector) {
            this.vector = vector;
        }

        public String toFeatureString() {
            return Joiner.on(",").join(vector);
        }
    }

    // The tree structure of plan
    static class OperatorWithFeatures extends TreeNode<OperatorWithFeatures> {
        int planNodeId;
        OperatorFeatures features;

        public static OperatorWithFeatures build(int planNodeId, OperatorFeatures features) {
            OperatorWithFeatures res = new OperatorWithFeatures();
            res.planNodeId = planNodeId;
            res.features = features;
            return res;
        }

        public List<Integer> toVector() {
            return features.toVector();
        }
    }

    static class PlanTreeBuilder {

    }

    // TODO: build specific features for operator
    public static class OperatorFeatures {
        OperatorType opType;
        CostEstimate cost;
        Statistics stats;

        static OperatorFeatures build(OperatorType type, CostEstimate cost, Statistics stats) {
            OperatorFeatures res = new OperatorFeatures();
            res.opType = type;
            res.cost = cost;
            res.stats = stats;
            return res;
        }

        public List<Integer> toVector() {
            List<Integer> res = Lists.newArrayList();
            res.add((int) cost.getMemoryCost());
            res.add((int) stats.getOutputRowCount());
            res.add((int) stats.getAvgRowSize());

            return res;
        }

        public static int numFeatures() {
            return 3;
        }
    }

    static class JoinOperatorFeatures extends OperatorWithFeatures {

    }

    static class AggOperatorFeatures extends OperatorWithFeatures {

    }

    static class Extractor extends OptExpressionVisitor<OperatorWithFeatures, PlanTreeBuilder> {

        @Override
        public OperatorWithFeatures visit(OptExpression optExpression, PlanTreeBuilder context) {
            OperatorType opType = optExpression.getOp().getOpType();
            Statistics stats = optExpression.getStatistics();
            CostEstimate cost = CostModel.calculateCostEstimate(new ExpressionContext(optExpression));

            OperatorFeatures features = OperatorFeatures.build(opType, cost, stats);
            OperatorWithFeatures node = OperatorWithFeatures.build(optExpression.getOp().getPlanNodeId(), features);

            // recursive visit
            for (var child : optExpression.getInputs()) {
                OperatorWithFeatures childNode = visit(child, context);
                node.addChild(childNode);
            }

            return node;
        }

    }
}
