"""The statistic each part of each experiment's rule reads, held in code (PLAN.md D116 hardening), so a source edit
cannot move a rule to another statistic of the same file: the regret coefficient's main rule onto the slope, a
slope condition onto the regret coefficient, epv-E1's overall difference onto one time bucket's.

Keyed by experiment, then "part|stage" (part empty for an experiment with one rule), with the main rule's paths
(value | lower end | upper end, "-" where the file states none) and each condition's, keyed by its met flag or else by
its comparison and bar. Written from the reviewed source at v4-research 5e7bf412 and read by
export.research_methods; a new experiment adds its entry here, by hand, in the change that adds its result.
"""

RULE_STATISTICS = {
    'decisions-E2': {
        '|development': {
            "main": 'metrics.new.regret_coef.point | metrics.new.regret_coef.lo | metrics.new.regret_coef.hi',
            "conditions": {
                'at_least 0.8':
                    'metrics.new.slope.point | metrics.new.slope.lo | metrics.new.slope.hi',
                'at_most 1.2':
                    'metrics.new.slope.point | metrics.new.slope.lo | metrics.new.slope.hi',
                'decision.checks.paired_difference_above_zero':
                    'metrics.paired_regret_coef_new_minus_frozen.point | metrics.paired_regret_coef_new_minus_frozen.lo | metrics.paired_regret_coef_new_minus_frozen.hi',
            },
        },
    },
    'embeddings-E1': {
        '|test': {
            "main": 'scored.rates_with_game_bootstrap.pca.value | scored.rates_with_game_bootstrap.pca.lo | scored.rates_with_game_bootstrap.pca.hi',
            "conditions": {
                'above scored.rule.position_upper_end':
                    'scored.rates_with_game_bootstrap.pca.value | scored.rates_with_game_bootstrap.pca.lo | scored.rates_with_game_bootstrap.pca.hi',
            },
        },
    },
    'embeddings-E2': {
        '|development': {
            "main": 'verdict.difference | verdict.lo | verdict.hi',
            "conditions": {
                'verdict.size_at_least_threshold':
                    'verdict.difference | verdict.lo | verdict.hi',
            },
        },
    },
    'epv-E1': {
        '|development': {
            "main": 'overall.baseline_minus_e1.point | overall.baseline_minus_e1.ci95.0 | overall.baseline_minus_e1.ci95.1',
            "conditions": {
                'at_least 0.9':
                    'verdict.slope_0_4_e1 | - | -',
                'at_most 1.1':
                    'verdict.slope_0_4_e1 | - | -',
            },
        },
    },
    'genai-E1': {
        'H1: the checker does not separate a template from the model|test': {
            "main": '- | scouting.bootstrap_by_player.differences.haiku minus template: pass_rate.0 | scouting.bootstrap_by_player.differences.haiku minus template: pass_rate.1',
            "conditions": {
            },
        },
        'H2: the model writes with more variety|test': {
            "main": '- | scouting.bootstrap_by_player.differences.haiku minus template: frame_variety.0 | scouting.bootstrap_by_player.differences.haiku minus template: frame_variety.1',
            "conditions": {
            },
        },
    },
    'genai-E2': {
        '0.5B H1 possession: training helps|test': {
            "main": 'hypotheses.H1 possession: a1-epoch1 minus a0.point | hypotheses.H1 possession: a1-epoch1 minus a0.interval.0 | hypotheses.H1 possession: a1-epoch1 minus a0.interval.1',
            "conditions": {
                'at_least 0.1':
                    'hypotheses.H1 possession: a1-epoch1 minus a0.point | hypotheses.H1 possession: a1-epoch1 minus a0.interval.0 | hypotheses.H1 possession: a1-epoch1 minus a0.interval.1',
            },
        },
        '0.5B H1 scouting: training helps|test': {
            "main": 'hypotheses.H1 scouting: a1-epoch1 minus a0.point | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.0 | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.1',
            "conditions": {
                'at_least 0.1':
                    'hypotheses.H1 scouting: a1-epoch1 minus a0.point | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.0 | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.1',
            },
        },
        '0.5B H2 possession: rejection sampling|test': {
            "main": 'hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.point | hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.interval.0 | hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '0.5B H2 scouting: rejection sampling|test': {
            "main": 'hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.point | hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.interval.0 | hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '0.5B H3 scouting: near the API model|test': {
            "main": 'hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.point | hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.interval.0 | hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.interval.1',
            "conditions": {
            },
        },
        '0.5B H4 possession: constrained decoding, pass rate|test': {
            "main": 'hypotheses.H4 possession: a3 minus a1-epoch1.point | hypotheses.H4 possession: a3 minus a1-epoch1.interval.0 | hypotheses.H4 possession: a3 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '0.5B H4 scouting: constrained decoding, pass rate|test': {
            "main": 'hypotheses.H4 scouting: a3 minus a1-epoch1.point | hypotheses.H4 scouting: a3 minus a1-epoch1.interval.0 | hypotheses.H4 scouting: a3 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '1.5B H1 possession: training helps|test': {
            "main": 'hypotheses.H1 possession: a1-epoch1 minus a0.point | hypotheses.H1 possession: a1-epoch1 minus a0.interval.0 | hypotheses.H1 possession: a1-epoch1 minus a0.interval.1',
            "conditions": {
                'at_least 0.1':
                    'hypotheses.H1 possession: a1-epoch1 minus a0.point | hypotheses.H1 possession: a1-epoch1 minus a0.interval.0 | hypotheses.H1 possession: a1-epoch1 minus a0.interval.1',
            },
        },
        '1.5B H1 scouting: training helps|test': {
            "main": 'hypotheses.H1 scouting: a1-epoch1 minus a0.point | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.0 | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.1',
            "conditions": {
                'at_least 0.1':
                    'hypotheses.H1 scouting: a1-epoch1 minus a0.point | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.0 | hypotheses.H1 scouting: a1-epoch1 minus a0.interval.1',
            },
        },
        '1.5B H2 possession: rejection sampling|test': {
            "main": 'hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.point | hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.interval.0 | hypotheses.H2 possession: a2-r1-epoch1 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '1.5B H2 scouting: rejection sampling|test': {
            "main": 'hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.point | hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.interval.0 | hypotheses.H2 scouting: a2-r1-epoch1 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '1.5B H3 scouting: near the API model|test': {
            "main": 'hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.point | hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.interval.0 | hypotheses.H3 scouting: a1-epoch1+repair minus haiku_cache.interval.1',
            "conditions": {
            },
        },
        '1.5B H4 possession: constrained decoding, pass rate|test': {
            "main": 'hypotheses.H4 possession: a3 minus a1-epoch1.point | hypotheses.H4 possession: a3 minus a1-epoch1.interval.0 | hypotheses.H4 possession: a3 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
        '1.5B H4 scouting: constrained decoding, pass rate|test': {
            "main": 'hypotheses.H4 scouting: a3 minus a1-epoch1.point | hypotheses.H4 scouting: a3 minus a1-epoch1.interval.0 | hypotheses.H4 scouting: a3 minus a1-epoch1.interval.1',
            "conditions": {
            },
        },
    },
    'ghosting-E1': {
        '|development': {
            "main": 'paired_ft.1.0.rollout_minus_mlp.diff | paired_ft.1.0.rollout_minus_mlp.lo | paired_ft.1.0.rollout_minus_mlp.hi',
            "conditions": {
                'above sign_test.rollout.n_neg':
                    'sign_test.rollout.n_pos | - | -',
                'at_most -0.1':
                    'paired_ft.1.0.rollout_minus_mlp.diff | - | -',
                'below 0.05':
                    'sign_test.rollout.p_value | - | -',
                'decision.guard_02s.ok':
                    'paired_ft.0.2.rollout_minus_mlp.diff | paired_ft.0.2.rollout_minus_mlp.lo | paired_ft.0.2.rollout_minus_mlp.hi',
            },
        },
        '|test': {
            "main": 'scored.1.0.rollout_minus_mlp.diff | scored.1.0.rollout_minus_mlp.lo | scored.1.0.rollout_minus_mlp.hi',
            "conditions": {
            },
        },
    },
    'gravity-E1': {
        '|development': {
            "main": 'bootstrap.intervals.candidate_minus_baseline.value | bootstrap.intervals.candidate_minus_baseline.lo | bootstrap.intervals.candidate_minus_baseline.hi',
            "conditions": {
                'decision.effect_ok':
                    'effect_per_sd_gravity.sum.value | effect_per_sd_gravity.sum.lo | effect_per_sd_gravity.sum.hi',
            },
        },
    },
    'lineups-E1': {
        '|test': {
            "main": 'whole_game.paired.rapm_minus_plus_minus.point | whole_game.paired.rapm_minus_plus_minus.ci95.0 | whole_game.paired.rapm_minus_plus_minus.ci95.1',
            "conditions": {
                'verdict.r2_above_zero_with_interval':
                    'whole_game.r2_vs_floor.rapm.point | whole_game.r2_vs_floor.rapm.ci95.0 | whole_game.r2_vs_floor.rapm.ci95.1',
            },
        },
    },
    'lineups-E2': {
        '|test': {
            "main": 'scored.results.paired_lrapm_minus_rapm_synergy.point | scored.results.paired_lrapm_minus_rapm_synergy.ci95.0 | scored.results.paired_lrapm_minus_rapm_synergy.ci95.1',
            "conditions": {
                'scored.results.verdict.against_rapm_synergy.d_at_most_minus_0_01':
                    'scored.results.relative_improvement_vs_rapm_synergy.point | scored.results.relative_improvement_vs_rapm_synergy.ci95.0 | scored.results.relative_improvement_vs_rapm_synergy.ci95.1',
                'scored.results.verdict.against_raw.d_at_most_minus_0_01':
                    'scored.results.relative_improvement_vs_raw.lrapm.point | scored.results.relative_improvement_vs_raw.lrapm.ci95.0 | scored.results.relative_improvement_vs_raw.lrapm.ci95.1',
                'scored.results.verdict.against_raw.diff_ci_below_zero':
                    'scored.results.paired_minus_raw.lrapm.point | scored.results.paired_minus_raw.lrapm.ci95.0 | scored.results.paired_minus_raw.lrapm.ci95.1',
            },
        },
    },
    'matchups-E1': {
        "Part (a): who takes the possession's last shot|development": {
            "main": 'part_a.defender_minus_no_defender.point | part_a.defender_minus_no_defender.ci95.0 | part_a.defender_minus_no_defender.ci95.1',
            "conditions": {
            },
        },
        "Part (a): who takes the possession's last shot|test": {
            "main": 'scored.parts.a.defender_minus_no_defender.point | scored.parts.a.defender_minus_no_defender.ci95.0 | scored.parts.a.defender_minus_no_defender.ci95.1',
            "conditions": {
            },
        },
        'Part (b): make given xFG|development': {
            "main": 'part_b.defender_minus_no_defender.point | part_b.defender_minus_no_defender.ci95.0 | part_b.defender_minus_no_defender.ci95.1',
            "conditions": {
            },
        },
        'Part (b): make given xFG|test': {
            "main": 'scored.parts.b.defender_minus_no_defender.point | scored.parts.b.defender_minus_no_defender.ci95.0 | scored.parts.b.defender_minus_no_defender.ci95.1',
            "conditions": {
            },
        },
    },
    'matchups-E2': {
        '|development': {
            "main": 'hmm_minus_hungarian.point | hmm_minus_hungarian.ci95.0 | hmm_minus_hungarian.ci95.1',
            "conditions": {
            },
        },
    },
    'winprob-E1': {
        '|development': {
            "main": 'pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.point | pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.lo | pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.hi',
            "conditions": {
                '{verdict}.size_at_least_threshold':
                    'pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.point | pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.lo | pooled_forward.paired.e1_minus_baseline.mean_bucket_brier.hi',
            },
        },
        '|test': {
            "main": 'scored.results.paired.e1_minus_baseline.mean_bucket_brier.point | scored.results.paired.e1_minus_baseline.mean_bucket_brier.lo | scored.results.paired.e1_minus_baseline.mean_bucket_brier.hi',
            "conditions": {
                '{verdict}.size_at_least_threshold':
                    'scored.results.paired.e1_minus_baseline.mean_bucket_brier.point | scored.results.paired.e1_minus_baseline.mean_bucket_brier.lo | scored.results.paired.e1_minus_baseline.mean_bucket_brier.hi',
            },
        },
    },
    'winprob-E2': {
        '|development': {
            "main": 'pooled_nested_forward.paired.e2_minus_e1.mean_bucket_ece.point | pooled_nested_forward.paired.e2_minus_e1.mean_bucket_ece.lo | pooled_nested_forward.paired.e2_minus_e1.mean_bucket_ece.hi',
            "conditions": {
                '{verdict}.brier_guard_holds':
                    'pooled_nested_forward.paired.e2_minus_e1.mean_bucket_brier.point | pooled_nested_forward.paired.e2_minus_e1.mean_bucket_brier.lo | pooled_nested_forward.paired.e2_minus_e1.mean_bucket_brier.hi',
            },
        },
    },
    'xfg-E1': {
        '|development': {
            "main": 'paired_cvap_minus_iso_shot_weighted.diff | paired_cvap_minus_iso_shot_weighted.lo | paired_cvap_minus_iso_shot_weighted.hi',
            "conditions": {
            },
        },
        '|test': {
            "main": 'paired_cvap_minus_iso_shot_weighted.diff | paired_cvap_minus_iso_shot_weighted.lo | paired_cvap_minus_iso_shot_weighted.hi',
            "conditions": {
            },
        },
    },
    'xfg-E2': {
        '|development': {
            "main": 'paired_new_minus_baseline.diff | paired_new_minus_baseline.lo | paired_new_minus_baseline.hi',
            "conditions": {
                'at_most verdict.threshold':
                    'paired_new_minus_baseline.diff | paired_new_minus_baseline.lo | paired_new_minus_baseline.hi',
            },
        },
        '|test': {
            "main": 'scored.paired_new_minus_baseline.diff | scored.paired_new_minus_baseline.lo | scored.paired_new_minus_baseline.hi',
            "conditions": {
            },
        },
    },
}
