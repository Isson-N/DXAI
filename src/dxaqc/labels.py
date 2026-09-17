"""Закрытые списки значений из разъяснений организатора (вопросы 6 и 15)."""

REGION_SPINE = "Поясничный отдел позвоночника"
REGION_HIP = "Проксимальный отдел бедра"

V_POSITIONING = "Некорректная укладка"
V_AXIS = "Не выравнена ось позвоночника"
V_FOREIGN = "Присутствуют посторонние предметы"
V_ROI = "Некорректная область интереса"

# порядок записи нарушений в violation_type
VIOLATIONS = {
    REGION_SPINE: (V_POSITIONING, V_AXIS, V_FOREIGN),
    REGION_HIP: (V_POSITIONING, V_ROI),
}

STATUS_OK = "Success"
STATUS_FAIL = "Failure"
