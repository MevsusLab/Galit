"""ГАЛИТ -- сквозная демонстрация: диагностика фонда и экономика.

Запуск:  python demo.py
Графики: python demo.py --plots
"""
from __future__ import annotations

import sys

from galit.demo_scenarios import DEMO_LABELS, run_competition_scenarios
from galit.economics import (
    compute_effect,
    default_assumptions,
    scenario_bounds,
    tornado,
    unknowns,
)
from galit.integrated import rank_wells
from galit.scale import WaterAnalysis, lsi_langelier, stiff_davis_index_checked
from galit.synthetic import make_fund

NAMES = {"halite": "галит", "calcite": "кальцит",
         "wax": "АСПО", "corrosion": "коррозия"}


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def demo_index_applicability() -> None:
    """Главный тезис: для этих вод неприменимы ОБА простых индекса."""
    hr("1. ГРАНИЦЫ ПРИМЕНИМОСТИ ИНДЕКСОВ")

    brine = WaterAnalysis(
        ions_mg_l={"Na": 92_000.0, "Cl": 175_000.0, "Ca": 22_000.0,
                   "Mg": 2_400.0, "HCO3": 120.0, "SO4": 300.0},
        ph=6.1, t_c=45.0, p_pa=4e6,
    )
    fresh = WaterAnalysis(
        ions_mg_l={"Na": 50.0, "Cl": 60.0, "Ca": 80.0, "HCO3": 150.0},
        ph=7.5, t_c=20.0, p_pa=1e5,
    )

    for label, w in (("Пресная вода", fresh),
                     ("Рассол Припятского прогиба", brine)):
        lsi, lsi_warn = lsi_langelier(w)
        sdsi, sd_warns = stiff_davis_index_checked(w)
        print(f"\n{label}:")
        print(f"  TDS = {w.tds_mg_l/1000:8.1f} г/л     "
              f"ионная сила = {w.ionic_strength:.2f} моль/л")
        print(f"  LSI        = {lsi:+7.2f}   "
              f"{'ВНЕ ОБЛАСТИ' if lsi_warn else 'применим'}")
        print(f"  Stiff-Davis= {sdsi:+7.2f}   "
              f"{'ВНЕ ОБЛАСТИ' if sd_warns else 'применим'}")
        if lsi_warn:
            print(f"    ! {lsi_warn}")
        for wn in sd_warns:
            print(f"    ! {wn}")

    lsi, _ = lsi_langelier(brine)
    sdsi, _ = stiff_davis_index_checked(brine)
    print(f"\n  Расхождение LSI и Stiff-Davis на рассоле: "
          f"{lsi - sdsi:.2f} лог-единиц.")
    print("  LSI систематически ЗАВЫШАЕТ склонность к отложению:")
    print("  высокая ионная сила повышает растворимость, чего он не учитывает.")


def demo_single_well() -> None:
    """Разбор одной скважины: профили, глубина АСПО, рекомендация."""
    hr("2. ДИАГНОСТИКА ОДНОЙ СКВАЖИНЫ")

    fund = make_fund(40)
    ranked = rank_wells(fund)
    r = ranked[0]

    print(f"\nСкважина: {r.well}   (наивысший риск в фонде)")
    print(f"Интегральный риск: {r.integrated_risk:.3f}   "
          f"доминирующий механизм: {NAMES[r.dominant]}")

    print("\n  Профиль в стволе:")
    print(f"    {'глубина, м':>12} {'T потока, C':>12} {'WAT, C':>9} "
          f"{'P, МПа':>9}")
    n = len(r.depths)
    for i in range(0, n, max(n // 8, 1)):
        print(f"    {r.depths[i]:12.0f} {r.temps[i]:12.1f} "
              f"{r.wat_profile[i]:9.1f} {r.pressures[i]/1e6:9.2f}")

    print("\n  Механизмы (шкала 0..1):")
    for k, v in sorted(r.severity.items(), key=lambda x: -x[1]):
        bar = "#" * int(v * 40)
        print(f"    {NAMES[k]:>10}  {v:5.3f}  {bar}")

    if r.wax_onset_m is not None:
        print(f"\n  Глубина начала АСПО: {r.wax_onset_m:.0f} м")
        print("    (пересечение кривых T потока и температуры насыщения)")
    else:
        print("\n  АСПО: весь ствол горячее WAT, отложений нет")

    print(f"\n  Соли:  SI(галит) = {r.scale['si_halite']:+.3f}   "
          f"m = {r.scale['m_nacl']:.2f} при насыщении {r.scale['m_sat']:.2f} моль/кг")
    print(f"         SI(кальцит) = {r.scale['si_calcite']:+.3f}   "
          f"TDS = {r.scale['tds_g_l']:.0f} г/л")
    if r.scale["halite_kg_m3"] > 0:
        print(f"         выпадение галита: {r.scale['halite_kg_m3']:.2f} кг/м3 воды")

    c = r.corrosion
    print(f"\n  Коррозия: {c['rate_mm_yr']:.4f} мм/год ({c['category']}), "
          f"лимитирует {c['limiting']}")
    print(f"            fCO2 = {c['f_co2_bar']:.3f} бар, "
          f"pH = {c['ph_actual']:.2f}")

    print(f"\n  РЕКОМЕНДАЦИЯ: {r.recommendation}")
    for w in r.warnings:
        print(f"  ! {w}")


def demo_fund_ranking() -> None:
    """Ранжирование фонда -- то, ради чего продукт и нужен."""
    hr("3. РАНЖИРОВАНИЕ ФОНДА (40 синтетических скважин)")

    fund = make_fund(40)
    lift = {c.name: c.lift_type for c in fund}
    ranked = rank_wells(fund)

    print(f"\n{'#':>3} {'скважина':<22} {'подъём':<7} {'риск':>6} "
          f"{'домин.':<10} {'АСПО, м':>8}")
    print("-" * 78)
    for i, r in enumerate(ranked[:15], 1):
        onset = f"{r.wax_onset_m:.0f}" if r.wax_onset_m is not None else "-"
        case_lift = lift[r.well]
        print(f"{i:>3} {r.well:<22} {case_lift:<7} {r.integrated_risk:6.3f} "
              f"{NAMES[r.dominant]:<10} {onset:>8}")

    print(f"\n  ... ещё {len(ranked) - 15} скважин ниже по риску")

    counts: dict[str, int] = {}
    for r in ranked:
        counts[r.dominant] = counts.get(r.dominant, 0) + 1
    print("\n  Структура фонда по доминирующему механизму:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {NAMES[k]:>10}: {v:3d} скв.  {'#' * v}")

    high = [r for r in ranked if r.integrated_risk > 0.35]
    print(f"\n  Скважин с риском > 0.35: {len(high)} из {len(ranked)} "
          f"({len(high)/len(ranked)*100:.0f} %) -- приоритет обработки")


def demo_competition_scenarios() -> None:
    """Отдельный конкурсный режим; unbiased synthetic fund не изменяется."""
    hr("КОНКУРСНЫЙ НАБОР АРХЕТИПОВ")
    print("  " + " / ".join(label.upper() for label in DEMO_LABELS))
    print("  Учебные входы; результаты заново рассчитаны ядром diagnose.")
    for item in run_competition_scenarios():
        r = item.diagnosis
        focus = item.scenario.educational_focus
        print(f"\n  {item.scenario.title} [{item.scenario.key}]")
        print(f"    учебный акцент: {focus}; фактический dominant: {r.dominant}")
        print("    severity: " + ", ".join(
            f"{key}={value:.3f}" for key, value in r.severity.items()
        ))
        print(f"    риск={r.integrated_risk:.3f}; {item.scenario.interpretation_note}")
        if item.co2_sensitivity:
            print("    CO2 sensitivity (доля CO2 -> коррозия, мм/год): " + ", ".join(
                f"{p.co2_mol_frac:.3f}->{p.corrosion_rate_mm_yr:.3f}"
                for p in item.co2_sensitivity
            ))
        if item.counterfactual:
            cf = item.counterfactual
            print(f"    counterfactual: {cf.action}")
            print(f"      до {cf.before.corrosion['rate_mm_yr']:.3f} -> "
                  f"после {cf.after.corrosion['rate_mm_yr']:.3f} мм/год")


def demo_economics() -> None:
    """Экономика с допущениями и чувствительностью."""
    hr("4. ИСТОРИЧЕСКИЙ СЦЕНАРНЫЙ ENVELOPE — НЕ ПРОГНОЗ")
    print("  NOT FIELD VALIDATED / NOT A KPI. Для BYN-эффекта нужны ставки заказчика.")

    a = default_assumptions()
    res = compute_effect(a)

    print("\n  Разложение годового эффекта:")
    for k, v in res.breakdown.items():
        share = v / res.total * 100 if res.total else 0
        print(f"    {k:>12}: {v:>14,.0f} BYN/год  ({share:4.1f} %)")
    print(f"    {'ИТОГО':>12}: {res.total:>14,.0f} BYN/год")

    print("\n  Промежуточные величины:")
    for k, v in res.detail.items():
        print(f"    {k:>28}: {v:>12,.1f}")

    print("\n  Сценарии (варьируются только ЭФФЕКТЫ, не параметры фонда):")
    for name, val in scenario_bounds(a).items():
        print(f"    {name:>16}: {val:>14,.0f} BYN/год")

    print("\n  Чувствительность (tornado, топ-6 по влиянию):")
    print(f"    {'параметр':<32} {'при min':>13} {'при max':>13} {'размах':>13}")
    for name, lo, hi, span in tornado(a)[:6]:
        print(f"    {name:<32} {lo:>13,.0f} {hi:>13,.0f} {span:>13,.0f}")

    print("\n  ТРЕБУЕТ УТОЧНЕНИЯ У ЗАКАЗЧИКА (по убыванию влияния):")
    for i, u in enumerate(unknowns(a)[:8], 1):
        print(f"    {i}. {u.name} = {u.value:g} {u.unit}")
        print(f"       {u.source}")


def demo_plots() -> None:
    """Графики (требует matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hr("5. ГРАФИКИ")
    fund = make_fund(40)
    ranked = rank_wells(fund)
    r = ranked[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    ax = axes[0]
    ax.plot(r.temps, r.depths, label="T потока", lw=2)
    ax.plot(r.wat_profile, r.depths, "--", label="WAT (насыщение парафином)", lw=2)
    if r.wax_onset_m is not None:
        ax.axhline(r.wax_onset_m, color="red", ls=":", lw=2,
                   label=f"начало АСПО: {r.wax_onset_m:.0f} м")
    ax.invert_yaxis()
    ax.set_xlabel("температура, C")
    ax.set_ylabel("глубина, м")
    ax.set_title(f"Пересечение кривых\n{r.well}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    risks = [x.integrated_risk for x in ranked]
    colors = ["#c0392b" if v > 0.35 else "#7f8c8d" for v in risks]
    ax.barh(range(len(risks)), risks, color=colors)
    ax.invert_yaxis()
    ax.axvline(0.35, color="black", ls=":", lw=1)
    ax.set_xlabel("интегральный риск")
    ax.set_ylabel("скважины (ранжированы)")
    ax.set_title("Ранжирование фонда")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[2]
    mechs = ["halite", "calcite", "wax", "corrosion"]
    bottom = [0.0] * len(ranked[:20])
    for m in mechs:
        vals = [x.severity[m] for x in ranked[:20]]
        ax.bar(range(20), vals, bottom=bottom, label=NAMES[m])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("топ-20 скважин")
    ax.set_ylabel("вклад механизмов")
    ax.set_title("Структура осложнений")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = "galit_demo.png"
    plt.savefig(out, dpi=120)
    print(f"\n  Сохранено: {out}")


def main() -> None:
    print("=" * 78)
    print("ГАЛИТ -- интегрированный flow assurance".center(78))
    print("прогноз галита, кальцита, АСПО и CO2-коррозии".center(78))
    print("=" * 78)
    print("\nДАННЫЕ СИНТЕТИЧЕСКИЕ. Физика реальная, фонд смоделирован")
    print("по опубликованным характеристикам Припятского прогиба.")

    demo_index_applicability
    demo_single_well()
    demo_fund_ranking()
    if "--competition" in sys.argv:
        demo_competition_scenarios()
    demo_economics()

    if "--plots" in sys.argv:
        demo_plots()

    hr()
    print("Готово. Тесты: python -m pytest tests/ -q")


if __name__ == "__main__":
    main()
