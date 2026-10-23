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

    geometry=WellGeometry(
        depth_m=3200.0,
        tubing_id_m=0.062,
        inclination_deg=15.0,
    ),

    rate=ProductionRate(
        q_oil_m3d=8.0,
        q_water_m3d=72.0,
        gor_m3m3=65.0,
    ),

    fluid=FluidProperties(
        gamma_oil=0.86,
        gamma_gas=0.78,
        salinity_ppm=290_000.0,
    ),

    thermal=ThermalParams(
        t_surface_c=8.0,
        geothermal_grad=0.033,
        u_to=15.0,
        production_days=400.0,
    ),

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
        t_c=40.0,
        p_pa=5e6,
    ),

    wax=WaxProperties(
        wat_stock_tank_c=34.0,
        wax_content_pct=6.5,
    ),

    co2_mol_frac=0.012,
    inhibitor_efficiency=0.0,
    lift_type="ЭЦН",
    p_wellhead_pa=1.4e6,
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
