"""Минимальный пример: одна скважина, свои данные.

Скопируйте этот файл, подставьте свои цифры, запустите:
    python example_one_well.py
"""
from galit import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    diagnose,
)

case = WellCase(
    name="Речицкая 123",

    # --- 1. Конструкция скважины ---
    geometry=WellGeometry(
        depth_m=3200.0,          # глубина по стволу до забоя, м
        tubing_id_m=0.062,       # ВНУТРЕННИЙ диаметр НКТ, м (62 мм)
        inclination_deg=15.0,    # средний угол от вертикали, град
    ),

    # --- 2. Режим работы (замер на устье) ---
    rate=ProductionRate(
        q_oil_m3d=8.0,           # дебит нефти, м3/сут
        q_water_m3d=72.0,        # дебит воды, м3/сут  (обводнённость 90 %)
        gor_m3m3=65.0,           # газовый фактор, м3/м3
    ),

    # --- 3. Свойства флюидов ---
    fluid=FluidProperties(
        gamma_oil=0.86,          # отн. плотность нефти по воде
        gamma_gas=0.78,          # отн. плотность газа по воздуху
        salinity_ppm=290_000.0,  # минерализация воды, мг/л
    ),

    # --- 4. Теплофизика ---
    thermal=ThermalParams(
        t_surface_c=8.0,         # температура пород у поверхности, C
        geothermal_grad=0.033,   # геотермический градиент, К/м
        u_to=15.0,               # коэф. теплопередачи, Вт/(м2*К)
        production_days=400.0,   # сколько суток работает непрерывно
    ),

    # --- 5. Химанализ пластовой воды (мг/л) ---
    water=WaterAnalysis(
        ions_mg_l={
            "Na": 95_000.0,
            "Cl": 205_000.0,
            "Ca": 28_000.0,
            "Mg": 3_100.0,
            "K": 1_800.0,
            "HCO3": 130.0,
            "SO4": 250.0,
        },
        ph=6.0,
        t_c=40.0,                # температура отбора пробы
        p_pa=5e6,                # давление отбора пробы, Па
    ),

    # --- 6. Парафинистость (лабораторный замер) ---
    wax=WaxProperties(
        wat_stock_tank_c=34.0,   # WAT дегазированной нефти, C
        wax_content_pct=6.5,     # содержание парафина, % масс.
    ),

    # --- 7. Прочее ---
    co2_mol_frac=0.012,          # доля CO2 в попутном газе (1,2 %)
    inhibitor_efficiency=0.0,    # 0 = без ингибитора, 0.9 = 90 % защиты
    lift_type="ЭЦН",             # ЭЦН | ШГН | фонтан
    p_wellhead_pa=1.4e6,         # буферное давление, Па
)

r = diagnose(case)

NAMES = {"halite": "галит", "calcite": "кальцит",
         "wax": "АСПО", "corrosion": "коррозия"}

print(f"\nСкважина: {r.well}")
print(f"Интегральный риск: {r.integrated_risk:.3f}")
print(f"Доминирующий механизм: {NAMES[r.dominant]}\n")

print("Механизмы (0..1):")
for k, v in sorted(r.severity.items(), key=lambda x: -x[1]):
    print(f"  {NAMES[k]:>10}  {v:5.3f}  {'#' * int(v * 40)}")

print(f"\nУстье: T = {r.temps[0]:.1f} C, P = {r.pressures[0]/1e6:.2f} МПа")
print(f"Забой: T = {r.temps[-1]:.1f} C, P = {r.pressures[-1]/1e6:.2f} МПа")

if r.wax_onset_m is not None:
    print(f"Начало АСПО: {r.wax_onset_m:.0f} м от устья")
else:
    print("АСПО: отложений нет (весь ствол горячее WAT)")

print(f"\nSI(галит)   = {r.scale['si_halite']:+.3f}")
print(f"SI(кальцит) = {r.scale['si_calcite']:+.3f}")
if r.scale["halite_kg_m3"] > 0:
    print(f"Выпадение галита: {r.scale['halite_kg_m3']:.2f} кг/м3 воды")

print(f"\nКоррозия: {r.corrosion['rate_mm_yr']:.3f} мм/год "
      f"({r.corrosion['category']}), лимитирует {r.corrosion['limiting']}")

print(f"\nРЕКОМЕНДАЦИЯ: {r.recommendation}")
for w in r.warnings:
    print(f"  ! {w}")
