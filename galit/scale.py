"""Термодинамика солеотложений: галит (NaCl) и кальцит (CaCO3).

ВАЖНО О ГРАНИЦАХ ПРИМЕНИМОСТИ
-----------------------------
Индекс Ланжелье (LSI) для вод Припятского прогиба НЕПРИМЕНИМ.
ASTM D4582 предписывает Stiff-Davis выше 10 000 мг/л TDS; ASTM D3739
(Langelier) -- ниже. Пластовые воды Белоруснефти имеют плотность
1,18-1,25 г/см3, что соответствует ~250 000-330 000 мг/л TDS,
то есть на полтора порядка выше границы применимости LSI.

Поэтому здесь LSI реализован ТОЛЬКО как демонстрация ошибки
(см. lsi_langelier -- функция намеренно возвращает предупреждение),
а рабочим индексом является Stiff-Davis с расчётом ионной силы
по полному ионному составу.

Для галита published saturation index типа Oddo-Tomson НЕ СУЩЕСТВУЕТ:
NaCl слишком растворим, простой полином не работает. Строгий расчёт
требует модели Питцера (ScaleSoftPitzer, OLI MSE). Здесь реализован
скрининговый подход через отношение молальности к насыщающей
(Potter, Babcock & Brown 1977) с коэффициентом активности по
Питцеру в упрощённой однопараметрической форме.

Корреляции Oddo-Tomson НЕ реализованы сознательно: в открытом доступе
их коэффициенты присутствуют только в OCR-искажённом виде
(коэффициент при давлении читается как 6.33e-3 и как 6.33e-5,
разница в 100 раз -- при 3000 psi это 19 против 0.19 лог-единиц).
Кодировать формулу, которую нельзя проверить, нельзя.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .fluids import water_ionic_strength

# Молярные массы, г/моль
MW = {
    "Na": 22.990, "K": 39.098, "Ca": 40.078, "Mg": 24.305,
    "Ba": 137.327, "Sr": 87.62, "Fe": 55.845,
    "Cl": 35.453, "SO4": 96.06, "HCO3": 61.016, "CO3": 60.008,
}

# Граница применимости LSI по ASTM D4582 / D3739, мг/л
LSI_TDS_LIMIT = 10_000.0


@dataclass(frozen=True)
class WaterAnalysis:
    """Химический анализ пластовой воды.

    Концентрации ионов -- мг/л. Температура -- C. Давление -- Па.
    """
    ions_mg_l: dict[str, float]
    ph: float
    t_c: float
    p_pa: float

    @property
    def tds_mg_l(self) -> float:
        return sum(self.ions_mg_l.values())

    def molarity(self) -> dict[str, float]:
        """Концентрации, моль/л."""
        return {
            ion: mg / MW[ion] / 1000.0
            for ion, mg in self.ions_mg_l.items()
            if ion in MW
        }

    @property
    def ionic_strength(self) -> float:
        """Ионная сила, моль/л."""
        return water_ionic_strength(self.molarity())

    def water_kg_per_l(self) -> float:
        """Масса воды в литре раствора, кг/л."""
        rho_g_l = 1000.0 + 0.7 * self.tds_mg_l / 1000.0  # оценка
        return max((rho_g_l - self.tds_mg_l / 1000.0) / 1000.0, 0.1)

    def molality(self) -> dict[str, float]:
        """Концентрации, моль/кг H2O."""
        w = self.water_kg_per_l()
        return {ion: c / w for ion, c in self.molarity().items()}

    def molality_na_cl(self) -> tuple[float, float]:
        """Молальности Na+ и Cl- по отдельности, моль/кг H2O.

        Раздельно -- принципиально. Воды Припятского прогиба относятся
        к хлоркальциевому типу: хлорид присутствует в БОЛЬШОМ ИЗБЫТКЕ
        над натрием за счёт CaCl2 и MgCl2. Именно этот избыток
        (эффект одноимённого иона) высаливает галит.
        """
        mol = self.molality()
        return mol.get("Na", 0.0), mol.get("Cl", 0.0)

    def molality_nacl(self) -> float:
        """Эквивалентная молальность NaCl, моль/кг H2O.

        Среднее геометрическое sqrt(m_Na * m_Cl) -- та молальность
        чистого раствора NaCl, которая даёт то же произведение ионов.
        Используется как аргумент для коэффициента активности.

        ВНИМАНИЕ: НЕ min(Na, Cl). Лимитирующий ион здесь ни при чём --
        равновесие галита управляется ПРОИЗВЕДЕНИЕМ a(Na+)*a(Cl-),
        и избыток хлорида смещает его вправо, а не отбрасывается.
        """
        na, cl = self.molality_na_cl()
        return math.sqrt(max(na, 0.0) * max(cl, 0.0))


# --------------------------------------------------------------------------
# Галит
# --------------------------------------------------------------------------

def halite_saturation_molality(t_c: float) -> float:
    """Насыщающая молальность NaCl, моль/кг H2O.

    Potter R.W. II, Babcock S., Brown D.L. (1977),
    J. Research U.S. Geological Survey, v.5 no.3, p.389-395.

        wt% NaCl = 26.218 + 0.0072*t + 0.000106*t^2      (t в C)

    Заявленная точность +-0.05 wt%. Данные измерены при 148-425 C,
    уравнение валидно 0-800 C (ниже 148 C -- экстраполяция, хорошо
    согласующаяся с табличными данными).

    Проверено против справочных значений: 25 C -> 6.146 моль/кг.
    """
    w = 26.218 + 0.0072 * t_c + 0.000106 * t_c ** 2  # wt% NaCl
    return 1000.0 * w / (58.443 * (100.0 - w))


def halite_activity_coefficient(m_nacl: float, t_c: float,
                                ionic_strength: float | None = None) -> float:
    """Средний ионный коэффициент активности NaCl.

    Упрощённая форма Питцера (Pitzer & Mayorga 1973) с параметрами
    для NaCl при 25 C. Температурная зависимость параметров
    не учитывается -- это осознанное упрощение для скрининга,
    оно завышает gamma при высоких T.

    ionic_strength -- ионная сила ВСЕГО раствора, моль/кг. Если не
    задана, принимается I = m (чистый раствор NaCl). Для рассолов
    хлоркальциевого типа это существенно: CaCl2 и MgCl2 поднимают
    ионную силу заметно выше молальности NaCl, а gamma зависит
    именно от неё.

    Для строгого расчёта нужен ScaleSoftPitzer или OLI MSE.
    """
    if m_nacl <= 0:
        return 1.0
    # параметры Питцера для NaCl при 25 C
    b0, b1, c_phi = 0.0765, 0.2664, 0.00127
    a_phi = 0.392  # параметр Дебая-Хюккеля при 25 C
    b, alpha = 1.2, 2.0

    i = m_nacl if ionic_strength is None else max(ionic_strength, 1e-9)
    sqrt_i = math.sqrt(i)

    f_gamma = -a_phi * (
        sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i)
    )
    x = alpha * sqrt_i
    b_gamma = 2.0 * b0 + (2.0 * b1 / (alpha ** 2 * i)) * (
        1.0 - (1.0 + x + 0.5 * x ** 2) * math.exp(-x)
    )
    c_gamma = 1.5 * c_phi

    ln_gamma = f_gamma + m_nacl * b_gamma + m_nacl ** 2 * c_gamma
    return math.exp(ln_gamma)


def halite_saturation_index(water: WaterAnalysis) -> float:
    """Индекс насыщения по галиту.

        SI = log10( a(Na+)*a(Cl-) / Ksp ),   Ksp = (gamma_sat * m_sat)^2

    Ионное произведение считается по ФАКТИЧЕСКИМ молальностям Na и Cl,
    а не по лимитирующему иону: в хлоркальциевых рассолах Припятского
    прогиба Cl в избытке над Na, и этот избыток -- главная причина
    выпадения галита (эффект одноимённого иона). Коэффициент активности
    берётся при ионной силе всего раствора, включая вклад CaCl2/MgCl2.

    Ksp при насыщении относится к ЧИСТОМУ раствору NaCl, где I = m_sat.

    SI > 0 означает пересыщение.

    ВАЖНО: это скрининговый расчёт, а не строгая модель Питцера.
    Погрешность оценивается в +-0.15 лог-единиц при 25-90 C.
    Для проектных решений требуется калибровка по промысловым данным.
    """
    m_na, m_cl = water.molality_na_cl()
    if m_na <= 0 or m_cl <= 0:
        return -99.0

    m_sat = halite_saturation_molality(water.t_c)
    i_solution = water.ionic_strength / water.water_kg_per_l()  # моль/кг

    g = halite_activity_coefficient(math.sqrt(m_na * m_cl), water.t_c,
                                    ionic_strength=i_solution)
    g_sat = halite_activity_coefficient(m_sat, water.t_c)

    iap = (g * m_na) * (g * m_cl)
    ksp = (g_sat * m_sat) ** 2
    return math.log10(iap / ksp)


def halite_precipitation_potential(water: WaterAnalysis) -> float:
    """Масса выпадающего галита при достижении равновесия, кг/м3 воды.

    Галит выпадает в стехиометрии 1:1, поэтому осаждение x моль/кг
    снижает ОБА иона на x:

        (m_Na - x) * (m_Cl - x) = Ksp'          -- квадратное по x

    Решаем относительно меньшего корня. Простая разность
    (m - m_sat) здесь неверна: при избытке Cl натрий выпадает
    почти нацело, и линейная оценка сильно занижает массу.

    Отрицательное значение означает недосыщение -> возвращаем 0.
    """
    m_na, m_cl = water.molality_na_cl()
    if m_na <= 0 or m_cl <= 0:
        return 0.0

    m_sat = halite_saturation_molality(water.t_c)
    i_solution = water.ionic_strength / water.water_kg_per_l()
    g = halite_activity_coefficient(math.sqrt(m_na * m_cl), water.t_c,
                                    ionic_strength=i_solution)
    g_sat = halite_activity_coefficient(m_sat, water.t_c)

    # Ksp в терминах молальностей при текущем gamma
    ksp_m = (g_sat * m_sat / max(g, 1e-9)) ** 2
    if m_na * m_cl <= ksp_m:
        return 0.0

    # x^2 - (m_Na + m_Cl)*x + (m_Na*m_Cl - Ksp') = 0, меньший корень
    b = m_na + m_cl
    c = m_na * m_cl - ksp_m
    disc = max(b * b - 4.0 * c, 0.0)
    x = (b - math.sqrt(disc)) / 2.0
    x = min(max(x, 0.0), min(m_na, m_cl))

    water_kg_m3 = water.water_kg_per_l() * 1000.0
    return x * 58.443e-3 * water_kg_m3


# --------------------------------------------------------------------------
# Кальцит: Stiff-Davis (рабочий) и Langelier (для демонстрации ошибки)
# --------------------------------------------------------------------------

# Ионная сила, выше которой S&DSI выходит за пределы калибровки
# (Stiff & Davis 1952 калиброван на I = 0...4 моль/кг)
SDSI_IONIC_STRENGTH_LIMIT = 4.0

# Точка излома кусочного фита USBR
SDSI_FIT_BREAK = 1.2


def stiff_davis_k(ionic_strength: float, t_c: float) -> float:
    """Константа K в индексе Стиффа-Дэвиса.  T в градусах Цельсия.

    Оригинал (Stiff & Davis 1952, Pet. Trans. AIME 195, 213) содержит
    ТОЛЬКО номограмму (Fig. 1), без аналитической формы. Здесь
    воспроизведён кусочный фит USBR (WQeval documentation, стр. 16,
    Equations 13-14), построенный по графикам ASTM D4582-91:

        I < 1.2:  K = 2.022*exp[(ln I + 7.544)^2/102.60]
                      - 0.0002*T^2 + 0.00097*T + 0.262
        I > 1.2:  K = -0.1*I - 0.0002*T^2 - 0.00097*T + 3.887

    Коэффициенты сверены с первоисточником дословно, включая
    ПРОТИВОПОЛОЖНЫЕ знаки линейного члена по T в двух ветвях -- это
    не артефакт OCR, так в оригинале. Практически знак несущественен:
    вершина параболы первой ветви лежит при T = 2.4 C, то есть во всём
    промысловом диапазоне K убывает с температурой в обеих ветвях.

    ФОРМА КРИВОЙ: K НЕМОНОТОННА по ионной силе -- растёт до I = 1.2,
    затем убывает (наклон -0.1 на единицу I). Максимум физически
    реален: независимый термодинамический расчёт
    (K ~ pK2 - pKsp - log g_Ca - log g_HCO3 по Plummer & Busenberg 1982
    с коэффициентами активности по Дэвису) даёт максимум при I ~ 0.5.
    Причина немонотонности -- сам член Дебая-Хюккеля: при малых I
    доминирует -A*z^2*sqrt(I)/(1+sqrt(I)) (активности падают, K растёт),
    при больших -- линейный salting-out член +0.3*A*z^2*I
    (активности растут обратно, K падает).

    ИЗВЕСТНЫЕ ДЕФЕКТЫ ФИТА (документированы, не исправляются, потому
    что исправление увело бы нас от воспроизводимости стандарта):
      1. Ветви НЕ сшиваются при I = 1.2: разрыв от -0.11 (0 C)
         до -0.29 (90 C). При I около 1.2 индекс скачет на ~0.16
         при бесконечно малом изменении состава.
      2. Фит систематически ЗАВЫШЕН на +0.2...+0.6 относительно
         реальной номограммы, максимум ошибки при 40-70 C
         (сверено с оцифровкой Tian, Yan & Chen, IOP Conf. Ser.
         Earth Environ. Sci. 781 (2021) 022033, Table 1).
      3. Контрольный пример самого USBR (стр. 6) не воспроизводится
         их же формулой: напечатано S&DSI = 0.12, формула даёт -0.04.
      4. Существует перепечатка фита с ошибочным +0.1*I во второй
         ветви (Hamad & Kuwairi, doi:10.18280/mmep.110921, Table 8).
         Авторитетен вариант USBR с -0.1*I.
    """
    i = max(ionic_strength, 1e-6)
    if i < SDSI_FIT_BREAK:
        return (
            2.022 * math.exp((math.log(i) + 7.544) ** 2 / 102.60)
            - 0.0002 * t_c ** 2
            + 0.00097 * t_c
            + 0.262
        )
    return -0.1 * i - 0.0002 * t_c ** 2 - 0.00097 * t_c + 3.887


def stiff_davis_index(water: WaterAnalysis) -> float:
    """Индекс Стиффа-Дэвиса для CaCO3.

        S&DSI = pH - pCa - pAlk - K

    Применим при высокой ионной силе, в отличие от LSI, но НЕ безгранично
    (см. stiff_davis_index_checked -- она возвращает и предупреждение).
    Положительное значение -- склонность к отложению кальцита.
    """
    mol = water.molarity()
    ca = mol.get("Ca", 0.0)
    # щёлочность как эквивалент по HCO3 + 2*CO3
    alk = mol.get("HCO3", 0.0) + 2.0 * mol.get("CO3", 0.0)
    if ca <= 0 or alk <= 0:
        return -99.0

    p_ca = -math.log10(ca)
    p_alk = -math.log10(alk)
    k = stiff_davis_k(water.ionic_strength, water.t_c)
    return water.ph - p_ca - p_alk - k


def stiff_davis_index_checked(water: WaterAnalysis) -> tuple[float, list[str]]:
    """S&DSI вместе со списком предупреждений о границах применимости.

    Это принципиальный момент для рассолов Припятского прогиба:
    Stiff-Davis решает проблему LSI (высокая TDS), но сам калиброван
    лишь до I = 4 моль/кг, а у пластовых вод Белоруснефти
    (плотность 1,18-1,25 г/см3) ионная сила достигает 5-7.

    Иначе говоря, для этих вод НИ ОДИН из простых индексов формально
    не применим -- требуется модель Питцера. Продукт обязан сообщать
    об этом, а не выдавать число с видом уверенности.
    """
    warnings: list[str] = []
    i = water.ionic_strength

    if i > SDSI_IONIC_STRENGTH_LIMIT:
        warnings.append(
            f"ионная сила {i:.1f} моль/л выше предела калибровки "
            f"Stiff-Davis (I<={SDSI_IONIC_STRENGTH_LIMIT:.0f}): результат "
            f"экстраполяция, нужна модель Питцера (ScaleSoftPitzer/OLI MSE)"
        )
    if abs(i - SDSI_FIT_BREAK) < 0.15:
        warnings.append(
            f"ионная сила {i:.2f} близка к точке излома фита USBR "
            f"(I={SDSI_FIT_BREAK}): в ней разрыв ~0.16 лог-единиц"
        )
    if 40.0 <= water.t_c <= 70.0:
        warnings.append(
            "T в диапазоне 40-70 C, где фит USBR завышен на +0.2...+0.6 "
            "относительно номограммы ASTM D4582 -- прогноз консервативен"
        )
    if water.t_c > 90.0:
        warnings.append(
            f"T={water.t_c:.0f} C выше диапазона Stiff-Davis (0-90 C)"
        )

    return stiff_davis_index(water), warnings


def lsi_langelier(water: WaterAnalysis) -> tuple[float, str | None]:
    """Индекс Ланжелье. Возвращает (значение, предупреждение).

    Реализован ИСКЛЮЧИТЕЛЬНО для демонстрации того, насколько сильно
    он ошибается на высокоминерализованных водах. В продуктиве
    использовать stiff_davis_index.
    """
    mol = water.molarity()
    ca_mg_caco3 = mol.get("Ca", 0.0) * 100_090.0  # моль/л -> мг/л как CaCO3
    alk_mg_caco3 = (
        mol.get("HCO3", 0.0) + 2.0 * mol.get("CO3", 0.0)
    ) * 50_045.0
    if ca_mg_caco3 <= 0 or alk_mg_caco3 <= 0:
        return -99.0, "нет данных по Ca или щёлочности"

    a = (math.log10(max(water.tds_mg_l, 1.0)) - 1.0) / 10.0
    b = -13.12 * math.log10(water.t_c + 273.15) + 34.55
    c = math.log10(ca_mg_caco3) - 0.4
    d = math.log10(alk_mg_caco3)
    ph_s = (9.3 + a + b) - (c + d)

    warn = None
    if water.tds_mg_l > LSI_TDS_LIMIT:
        warn = (
            f"LSI неприменим: TDS {water.tds_mg_l / 1000:.0f} г/л превышает "
            f"границу ASTM D3739/D4582 ({LSI_TDS_LIMIT / 1000:.0f} г/л) "
            f"в {water.tds_mg_l / LSI_TDS_LIMIT:.0f} раз. "
            f"Используйте Stiff-Davis."
        )
    return water.ph - ph_s, warn


def scale_risk_profile(water: WaterAnalysis) -> dict[str, float]:
    """Сводка по обоим механизмам солеотложения."""
    si_halite = halite_saturation_index(water)
    si_calcite = stiff_davis_index(water)
    return {
        "si_halite": si_halite,
        "si_calcite": si_calcite,
        "halite_kg_m3": halite_precipitation_potential(water),
        "m_nacl": water.molality_nacl(),
        "m_sat": halite_saturation_molality(water.t_c),
        "ionic_strength": water.ionic_strength,
        "tds_g_l": water.tds_mg_l / 1000.0,
    }
