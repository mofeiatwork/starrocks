// This file is made available under Elastic License 2.0.
// This file is based on code available under the Apache license here:
//   https://github.com/apache/incubator-doris/blob/master/be/src/util/progress_updater.h

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <boost/cstdint.hpp>
#include <string>

namespace starrocks {

// Utility class to update progress.  This is split out so a different
// logging level can be set for these updates (GLOG_module)
// This class is thread safe.
// Example usage:
//   ProgressUpdater updater("Task", 100, 10);  // 100 items, print every 10%
//   updater.Update(15);  // 15 done, prints 15%
//   updater.Update(3);   // 18 done, doesn't print
//   update.Update(5);    // 23 done, prints 23%
class ProgressUpdater {
public:
    // label - label that is printed with each update.
    // max - maximum number of work items
    // update_period - how often the progress is spewed expressed as a percentage
    ProgressUpdater(std::string label, int64_t max, int update_period);

    ProgressUpdater();

    // Sets the GLOG level for this progress updater.  By default, this will use
    // 2 but objects can override it.
    void set_logging_level(int level) { _logging_level = level; }

    // 'delta' more of the work has been complete.  Will potentially output to
    // VLOG_PROGRESS
    void update(int64_t delta);

    // Returns if all tasks are done.
    bool done() const { return _num_complete >= _total; }

    int64_t total() const { return _total; }
    int64_t num_complete() const { return _num_complete; }

private:
    std::string _label;
    int _logging_level{2};
    int64_t _total{0};
    int _update_period{0};
    int64_t _num_complete{0};
    int _last_output_percentage{0};
};

} // namespace starrocks
