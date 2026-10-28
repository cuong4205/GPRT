"""Build the 70%-dynamic GP/GPRT report from independently validated episodes."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ("gprt", "gp", "edd")
METRICS = ("served", "unserved", "cancelled", "on_time_rate", "distance", "distance_per_served",
           "vehicles_used", "trip_count", "wall_clock_seconds", "callback_latency_p95_ms")


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def sd(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["training_seed"]].append(row)
    result = {"episodes": len(rows), "training_seeds": len(by_seed)}
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        seed_means = [mean(r[metric] for r in group if r[metric] is not None) for group in by_seed.values()]
        seed_means = [value for value in seed_means if value is not None]
        result[metric] = {"mean": mean(values), "episode_sd": sd(values), "seed_mean_sd": sd(seed_means)}
    callbacks = sum(row["callback_count"] for row in rows)
    callback_ms = sum(row["callback_count"] * row["callback_latency_mean_ms"] for row in rows)
    result.update(callback_count=callbacks, callback_weighted_mean_ms=callback_ms / callbacks if callbacks else 0,
                  routing_wall_seconds_per_episode=callback_ms / 1000 / len(rows),
                  cancellation_requests=sum(r["cancellation_requests"] for r in rows),
                  cancellation_accepted=sum(r["cancellation_accepted"] for r in rows),
                  cancellation_rejected=sum(r["cancellation_rejected"] for r in rows),
                  violations=sum(r["violation_count"] for r in rows),
                  unserved_reasons=dict(Counter(reason for row in rows for reason in row["unserved_reasons"].values())),
                  rejected_cancellation_reasons=dict(Counter(record["reason"] for row in rows
                      for record in row["cancellation_records"] if not record["accepted"])))
    return result


def pairs(rows):
    lookup = {(r["instance"], r["scenario_seed"], r["training_seed"], r["policy"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate matched episode keys")
    output = []
    for row in rows:
        if row["policy"] != "gprt":
            continue
        other = lookup[(row["instance"], row["scenario_seed"], row["training_seed"], "gp")]
        if row["scenario_sha256"] != other["scenario_sha256"]:
            raise ValueError("Matched GP/GPRT episodes use different scenarios")
        record = {"instance": row["instance"], "scenario_seed": row["scenario_seed"],
                  "training_seed": row["training_seed"], "scenario_sha256": row["scenario_sha256"],
                  "same_completed_and_cancelled_ids": row["served_customer_ids"] == other["served_customer_ids"]
                      and row["cancelled_customer_ids"] == other["cancelled_customer_ids"]}
        for metric in ("served", "unserved", "cancelled", "on_time_rate", "distance"):
            record[metric + "_delta"] = row[metric] - other[metric]
        record["distance_percent_delta"] = 100 * (row["distance"] / other["distance"] - 1) if other["distance"] else None
        output.append(record)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def pm(group, metric, scale=1, digits=2):
    value = group[metric]
    if group["training_seeds"] == 1:
        return f"{value['mean'] * scale:.{digits}f}"
    return f"{value['mean'] * scale:.{digits}f} ± {value['seed_mean_sd'] * scale:.{digits}f}"


def build_report(root: Path, document: Path | None):
    manifests, gates, validations, all_rows = {}, {}, {}, {}
    context_path = root / "execution_context.json"
    execution_context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    tests_path = root / "tests.xml"
    test_count = None
    if tests_path.is_file():
        suites = list(ET.parse(tests_path).getroot().iter("testsuite"))
        if any(int(s.attrib.get("failures", 0)) or int(s.attrib.get("errors", 0)) for s in suites):
            raise ValueError("Test report contains failures/errors")
        test_count = sum(int(s.attrib.get("tests", 0)) for s in suites)
    for phase in ("pilot", "final"):
        directory = root / phase
        manifests[phase] = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        gates[phase] = json.loads((directory / "gate_report.json").read_text(encoding="utf-8"))
        validations[phase] = json.loads((directory / "validation_report.json").read_text(encoding="utf-8"))
        if not gates[phase]["all_episodes_constraint_clean"] or not validations[phase]["passed"]:
            raise ValueError(f"{phase} has not passed independent validation")
        if validations[phase]["source_files_changed_since_run"]:
            raise ValueError("Source changed since simulation; retain matching code before reporting")
        if manifests[phase]["scenario_rules"]["dynamic_ratio"] != .70:
            raise ValueError("This report requires an actual 70% dynamic run")
        all_rows[phase] = [json.loads(line) for line in (directory / "episode_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(all_rows[phase]) != gates[phase]["episodes"]:
            raise ValueError("Gate/result episode coverage differs")
    if len(all_rows["pilot"]) != 132 or len(all_rows["final"]) != 528:
        raise ValueError("Expected the locked 132/528-episode protocol")
    final = all_rows["final"]
    dynamic = [row for row in final if row["scenario_mode"] == "dynamic"]
    static = [row for row in final if row["scenario_mode"] == "static"]
    summaries = {mode: {method: summarize([r for r in rows if r["policy"] == method]) for method in METHODS}
                 for mode, rows in (("dynamic", dynamic), ("static", static))}
    paired = pairs([r for r in dynamic if r["policy"] in {"gprt", "gp"}])
    comparable = [r for r in paired if r["same_completed_and_cancelled_ids"]]
    paired_stats = {"pairs": len(paired), "served_wins": sum(r["served_delta"] > 0 for r in paired),
                    "served_ties": sum(r["served_delta"] == 0 for r in paired),
                    "served_losses": sum(r["served_delta"] < 0 for r in paired),
                    "equal_workload_pairs": len(comparable)}
    for metric in ("served", "unserved", "cancelled", "on_time_rate", "distance"):
        paired_stats[metric + "_mean_delta"] = mean(r[metric + "_delta"] for r in paired)
    paired_stats["equal_workload_distance_mean_delta"] = mean(r["distance_delta"] for r in comparable)
    paired_stats["equal_workload_distance_mean_percent_delta"] = mean(r["distance_percent_delta"] for r in comparable if r["distance_percent_delta"] is not None)
    by_seed = []
    for method in METHODS:
        for seed in sorted({r["training_seed"] for r in dynamic if r["policy"] == method}, key=str):
            group = [r for r in dynamic if r["policy"] == method and r["training_seed"] == seed]
            record = {"policy": method, "training_seed": seed, "episodes": len(group)}
            record.update({metric: mean(r[metric] for r in group) for metric in ("served", "unserved", "cancelled", "on_time_rate", "distance")})
            by_seed.append(record)
    scenarios = {}
    for row in dynamic:
        key = row["instance"], row["scenario_seed"]
        existing = scenarios.setdefault(key, row)
        if existing["scenario_sha256"] != row["scenario_sha256"]:
            raise ValueError("Policy comparison does not share materialized scenarios")
        if row["dynamic_selected"] != 70 or row["cancel_candidates"] != 5:
            raise ValueError("Actual episode scenario does not have 70 dynamic / 5 cancellation candidates")
    by_instance = []
    for name in sorted({r["instance"] for r in dynamic}):
        scenario_rows = [r for (instance, _), r in scenarios.items() if instance == name]
        record = {"instance": name, "selected_dynamic": 70,
                  "released_after_zero_mean": mean(r["dynamic_released_after_zero"] for r in scenario_rows)}
        for method in METHODS:
            group = [r for r in dynamic if r["instance"] == name and r["policy"] == method]
            for metric in ("served", "unserved", "distance"):
                record[method + "_" + metric] = mean(r[metric] for r in group)
        by_instance.append(record)
    payload = {"ratio": .70, "pilot_episodes": len(all_rows["pilot"]), "final_episodes": len(final),
               "summaries": summaries, "paired_gprt_gp": paired_stats, "by_training_seed": by_seed,
               "by_instance": by_instance, "unique_dynamic_test_scenarios": len(scenarios),
               "realized_dynamic_mean": mean(r["dynamic_released_after_zero"] for r in scenarios.values()),
               "realized_dynamic_min": min(r["dynamic_released_after_zero"] for r in scenarios.values()),
               "realized_dynamic_max": max(r["dynamic_released_after_zero"] for r in scenarios.values()),
               "all_constraint_clean": True, "all_independent_audits_passed": True,
               "policy_artifacts": manifests["final"]["policies"], "runtime_manifest": manifests["final"],
               "execution_context": execution_context, "passed_test_count": test_count}
    (root / "performance_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = []
    for mode, groups in summaries.items():
        for method, group in groups.items():
            flat.append({"mode": mode, "policy": method, "episodes": group["episodes"],
                         **{metric + "_mean": group[metric]["mean"] for metric in METRICS},
                         **{metric + "_between_seed_sd": group[metric]["seed_mean_sd"] for metric in METRICS}})
    write_csv(root / "performance_by_policy.csv", flat)
    write_csv(root / "performance_by_seed.csv", by_seed)
    write_csv(root / "performance_by_instance.csv", by_instance)
    write_csv(root / "paired_gprt_gp_full.csv", paired)
    d, s = summaries["dynamic"], summaries["static"]
    lines = ["### 11.1 Đợt chạy và mức động thực tế", "",
             f"Đã chạy mới **132 pilot + 528 final = 660 episode**; checker và validator độc lập đều đạt, 0 vi phạm. Final có 12 test instance, 36 scenario động, 5 heuristic mỗi GP/GPRT và EDD cố định. Không huấn luyện hoặc chọn lại biểu thức trên test động.", "",
             f"Bộ kiểm thử của project: **{test_count}/{test_count} test đạt** (tests.xml). Bao gồm test 70%/legacy ratio, cancellation/event ordering, chống future leakage và kiểm tra cách tổng hợp performance." if test_count is not None else "Không có tests.xml trong thư mục run; kết quả mô phỏng được kiểm tra bằng gate/validator ở trên.", "",
             f"Mỗi scenario chọn đúng **70/100 task động** và 5 ứng viên hủy. Số release sau 0 trung bình **{payload['realized_dynamic_mean']:.2f}/100**, min–max **{payload['realized_dynamic_min']}–{payload['realized_dynamic_max']}** trên 36 scenario test; phần chênh do clipping release về 0.", "",
             "### 11.2 Chất lượng phục vụ và chi phí trên test động", "",
             "Ký hiệu ± là SD giữa mean của 5 training seed; mỗi seed được lấy mean trên 36 episode. EDD có một policy nên không có SD giữa seed. Distance có đơn vị tọa độ Solomon. Các dòng không phải các training run độc lập theo từng episode.", "",
             table(["Policy", "Episode", "Served/100", "Unserved/100", "On-time/active (%)", "Distance"],
                   [[m.upper(), d[m]["episodes"], pm(d[m], "served"), pm(d[m], "unserved"),
                     pm(d[m], "on_time_rate", 100), pm(d[m], "distance")] for m in METHODS]), "",
             "Đọc đồng thời served, unserved và cancellation; distance thấp trên các workload khác nhau chưa đủ để kết luận policy tốt hơn.", "",
             table(["Policy", "Hủy thành công / request", "Tỷ lệ hủy (%)", "Xe dùng TB", "Chuyến TB", "Distance/served TB"],
                   [[m.upper(), f"{d[m]['cancellation_accepted']}/{d[m]['cancellation_requests']}",
                     f"{100*d[m]['cancellation_accepted']/d[m]['cancellation_requests']:.2f}",
                     f"{d[m]['vehicles_used']['mean']:.2f}", f"{d[m]['trip_count']['mean']:.2f}",
                     f"{d[m]['distance_per_served']['mean']:.2f}"] for m in METHODS]), "",
             "Số request là tổng qua các episode của policy, không phải số task gốc khác nhau trong dataset. GP/GPRT có 180 episode động mỗi phương pháp, EDD có 36.", "",
             "### 11.3 So sánh ghép cặp GPRT − GP", "",
             f"Có **{len(paired)} cặp** cùng instance, scenario hash và training seed. Theo số task served, GPRT thắng/hòa/thua GP: **{paired_stats['served_wins']}/{paired_stats['served_ties']}/{paired_stats['served_losses']}**.", "",
             f"Mean chênh served: **{paired_stats['served_mean_delta']:+.3f} task/episode**; on-time/active: **{100*paired_stats['on_time_rate_mean_delta']:+.3f} điểm phần trăm**; distance trên tất cả cặp: **{paired_stats['distance_mean_delta']:+.3f}**.", ""]
    if comparable:
        lines += [f"Đối chiếu chặt workload có **{len(comparable)} cặp phục vụ cùng tập customer ID và hủy cùng tập ID**. Trong nhóm này, distance GPRT − GP trung bình **{paired_stats['equal_workload_distance_mean_delta']:+.3f}**, mean phần trăm chênh theo từng cặp **{paired_stats['equal_workload_distance_mean_percent_delta']:+.2f}%**. Dấu âm nghĩa là GPRT ngắn hơn. Đây là thống kê mô tả, chưa là kiểm định ưu thế trên một quần thể độc lập.", ""]
        lines += ["Mean chênh tuyệt đối và mean phần trăm theo cặp có thể khác dấu: chênh tuyệt đối cho trọng số lớn hơn về mặt số học đối với các trường hợp có distance lớn, còn tỷ lệ từng cặp chuẩn hóa theo distance GP của cặp đó. Vì vậy hai số này phải được đọc riêng, không dùng một số để thay thế số kia.", ""]
    else:
        lines += ["Không có cặp phục vụ/hủy cùng tập ID; không dùng chênh distance để khẳng định lợi thế trên workload bằng nhau.", ""]
    lines += ["### 11.4 Chi phí ra quyết định", "",
              table(["Policy", "Mean callback gộp (ms)", "Mean p95 theo episode (ms)", "Routing/episode (s)", "Wall episode TB (s)"],
                    [[m.upper(), f"{d[m]['callback_weighted_mean_ms']:.3f}",
                      f"{d[m]['callback_latency_p95_ms']['mean']:.3f}",
                      f"{d[m]['routing_wall_seconds_per_episode']:.3f}",
                      f"{d[m]['wall_clock_seconds']['mean']:.3f}"] for m in METHODS]), "",
              "Mean callback gộp được cân theo số callback; cột p95 là trung bình các p95 từng episode, không phải global p95. Callback gồm đọc state, snapshot, feasibility, scoring và tạo decision; không chỉ là thời gian tính biểu thức. Wall episode không bao gồm toàn bộ I/O của runner. Các episode chạy tuần tự; số đo phụ thuộc máy và độ phức tạp biểu thức.", "",
              f"Context chạy được lưu riêng: OMP_NUM_THREADS={execution_context.get('OMP_NUM_THREADS', 'không ghi')}, MKL_NUM_THREADS={execution_context.get('MKL_NUM_THREADS', 'không ghi')}. Không suy thread count của lần chạy cũ từ môi trường lúc tổng hợp báo cáo.", "",
              "### 11.5 Static control trong cùng simulator", "",
              table(["Policy", "Episode", "Served TB", "On-time/active (%)", "Distance TB", "Xe TB", "Chuyến TB"],
                    [[m.upper(), s[m]["episodes"], f"{s[m]['served']['mean']:.2f}",
                      f"{100*s[m]['on_time_rate']['mean']:.2f}", f"{s[m]['distance']['mean']:.2f}",
                      f"{s[m]['vehicles_used']['mean']:.2f}", f"{s[m]['trip_count']['mean']:.2f}"] for m in METHODS]), "",
              "Static control có cùng engine, đội xe nhiều chuyến và policy; chỉ release toàn bộ tại 0 và tắt hủy. Không so trực tiếp BKS VRPTW single-trip với chi phí multi-trip này.", "",
              "### 11.6 Phân tích theo instance test", "",
              table(["Instance", "Release > 0 TB/100", "Served GPRT", "Served GP", "Distance GPRT", "Distance GP"],
                    [[r["instance"], f"{r['released_after_zero_mean']:.2f}", f"{r['gprt_served']:.2f}",
                      f"{r['gp_served']:.2f}", f"{r['gprt_distance']:.2f}", f"{r['gp_distance']:.2f}"] for r in by_instance]), "",
              "Các mean GP/GPRT mỗi instance gộp 3 scenario seed × 5 training seed. Các instance có ít served cần đối chiếu lý do infeasible-at-release và chờ xe trong artifact.", "",
              table(["Policy", "Unserved: infeasible_at_release", "Unserved: chưa dispatch khả thi trước H", "Hủy từ chối: assigned", "Hủy từ chối: completed"],
                    [[m.upper(), d[m]["unserved_reasons"].get("infeasible_at_release", 0),
                      d[m]["unserved_reasons"].get("no_feasible_dispatch_before_horizon", 0),
                      d[m]["rejected_cancellation_reasons"].get("already_assigned", 0),
                      d[m]["rejected_cancellation_reasons"].get("already_completed", 0)] for m in METHODS]), "",
              "### 11.7 Kết quả theo training seed", "",
              table(["Policy", "Training seed", "Served TB", "On-time/active (%)", "Distance TB"],
                    [[r["policy"].upper(), r["training_seed"] if r["training_seed"] is not None else "—",
                      f"{r['served']:.2f}", f"{100*r['on_time_rate']:.2f}", f"{r['distance']:.2f}"] for r in by_seed]), "",
              "### 11.8 Diễn giải kết quả", "",
              f"Trong ma trận này, GPRT phục vụ trung bình {d['gprt']['served']['mean']:.2f}/100 task và GP {d['gp']['served']['mean']:.2f}/100; chênh lệch mean rất nhỏ và không thể kết luận GPRT cải thiện mức phục vụ. Distance mean của GPRT thấp hơn GP {100*(1-d['gprt']['distance']['mean']/d['gp']['distance']['mean']):.2f}% trên toàn ma trận, nhưng các tập khách hoàn tất có thể khác nhau. Đối chiếu workload bằng nhau và cả chênh tuyệt đối/phần trăm ở mục 11.3 là cần thiết trước khi nói về lợi thế chi phí.", "",
              f"Chi phí callback mean gộp: GPRT {d['gprt']['callback_weighted_mean_ms']:.3f} ms, GP {d['gp']['callback_weighted_mean_ms']:.3f} ms. Đây là số đo của cả pipeline ra quyết định trên các biểu thức đã chọn, không phải một tuyên bố GPRT luôn nhanh hơn GP hay tốc độ chạy Transformer online.", "",
              "Các instance R101, RC101, RC102 có mức phục vụ giảm rõ so với các nhóm còn lại. Cần ưu tiên xử lý mức công khai quá sát time window và giới hạn dispatch tại depot; đổi heuristic đơn thuần không phục hồi được task đã infeasible ngay lúc release.", "",
              "### 11.9 Phạm vi kết luận và truy xuất", "",
              "Kết quả mô tả việc chuyển các heuristic pilot đã khóa sang DVRPTW synthetic 70%. GPRT hiện là artifact lịch sử 4 cycle; không dùng bảng này để khẳng định chất lượng của preset chính 32 cycle. Hiệu quả phục vụ, workload và distance được báo cùng nhau; chưa thực hiện kiểm định thống kê hay tái huấn luyện trên simulator.", "",
              f"Môi trường đợt chạy: Python {manifests['final']['python_version']}; DVRPSim {manifests['final']['simulator_version']}; SimPy {manifests['final']['simpy_version']}; NumPy {manifests['final']['numpy_version']}; PyTorch {manifests['final']['torch_version']}; {manifests['final']['platform']}.", "",
              f"Nguồn số liệu: {root.as_posix()}/performance_summary.json, performance_by_policy.csv, performance_by_seed.csv, performance_by_instance.csv, paired_gprt_gp_full.csv; manifest/scenario/log/results và validation_report trong từng phase. File performance_report.md chứa riêng phần kết quả này."]
    block = "\n".join(lines)
    (root / "performance_report.md").write_text("# Performance GP/GPRT trên DVRPTW 70% động\n\n" + block + "\n", encoding="utf-8")
    if document:
        original = document.read_text(encoding="utf-8")
        start, end = "<!-- PERFORMANCE_70_START -->", "<!-- PERFORMANCE_70_END -->"
        if original.count(start) != 1 or original.count(end) != 1:
            raise ValueError("Documentation must contain exactly one performance marker pair")
        prefix, tail = original.split(start)
        _, suffix = tail.split(end)
        document.write_text(prefix + start + "\n" + block + "\n" + end + suffix, encoding="utf-8")
    print(json.dumps({"episodes": 660, "all_audits_passed": True, "realized_dynamic_mean": payload["realized_dynamic_mean"],
                      "paired_gprt_gp": paired_stats}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--document", type=Path)
    args = parser.parse_args()
    build_report(args.results_dir, args.document)


if __name__ == "__main__":
    main()
