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
        Map<OperatorType, SummarizedFeature> sumVector = Maps.newHashMap();
        sumByOperatorType(root, sumVector);
        FeatureVector result = new FeatureVector();

        // TODO: Add plan features
        // Add operator features
        for (int start = OperatorType.PHYSICAL.ordinal();
                start < OperatorType.SCALAR.ordinal();
                start++) {
            OperatorType opType = OperatorType.values()[start];
            SummarizedFeature vector = sumVector.get(opType);
            if (vector != null) {
                result.add(vector.finish());
            } else {
                result.add(SummarizedFeature.empty(opType));
            }
        }

        return result;
    }

    private static void sumByOperatorType(OperatorWithFeatures tree, Map<OperatorType, SummarizedFeature> sum) {
        List<Long> vector = tree.toVector();
        OperatorType opType = tree.features.opType;
        SummarizedFeature exist = sum.computeIfAbsent(opType, (x) -> new SummarizedFeature(opType));
        exist.summarize(tree);

        // recursive
        for (var child : tree.getChildren()) {
            sumByOperatorType(child, sum);
        }
    }

    private static class SummarizedFeature {
        OperatorType opType;
        int count = 0;
        FeatureVector vector;

        SummarizedFeature(OperatorType type) {
            this.opType = type;
        }

        public void summarize(OperatorWithFeatures node) {
            this.count++;
            if (this.vector == null) {
                this.vector = new FeatureVector(node.features.toVector());
            } else {
                // A + B => C
                List<Long> vector1 = node.features.toVector();
                for (int i = 0; i < vector.vector.size(); i++) {
                    this.vector.vector.set(i, this.vector.vector.get(i) + vector1.get(i));
                }
            }
        }

        public FeatureVector finish() {
            List<Long> result = Lists.newArrayList();
            result.add((long) opType.ordinal());
            result.add((long) count);
            if (vector != null) {
                result.addAll(vector.vector);
            }
            return new FeatureVector(result);
        }

        public static FeatureVector empty(OperatorType type) {
            List<Long> result = Lists.newArrayList();
            result.add((long) type.ordinal());
            result.add((long) 0);
            for (int i = 0; i < OperatorFeatures.numFeatures(); i++) {
                result.add(0L);
            }
            return new FeatureVector(result);
        }

        public static int numFeatures() {
            return OperatorFeatures.numFeatures() + 2;
        }
    }

    public static class FeatureVector {
        List<Long> vector = Lists.newArrayList();

        public FeatureVector() {
        }

        public FeatureVector(List<Long> vector) {
            this.vector = vector;
        }

        public String toFeatureString() {
            return Joiner.on(",").join(vector);
        }

        public void add(List<Long> vector) {
            this.vector.addAll(vector);
        }

        public void add(FeatureVector vector) {
            if (vector.vector != null) {
                this.vector.addAll(vector.vector);
            }
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

        public List<Long> toVector() {
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

        public List<Long> toVector() {
            List<Long> res = Lists.newArrayList();
            // TODO: remove this feature, which has no impact to the model
            res.add((long) cost.getMemoryCost());
            res.add((long) stats.getOutputRowCount());

            return res;
        }

        public static int numFeatures() {
            return 2;
        }
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
