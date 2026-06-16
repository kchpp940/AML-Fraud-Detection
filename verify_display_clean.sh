#!/bin/bash
# 验证：模板和 Streamlit 渲染函数中"不判断" process_status/risk_level/prediction_code
# （即不出现 {% if ... == ... %} 这样的模式或 Python 的 ==/!= 判断）。
# 仅"读取展示字段"是允许的（如 row.risk_level_label）。
cd /Users/pkcha/AML-Fraud-Detection

set -e
FAIL=0

# 禁止模式：对这些字段做任何状态判断（==/!=/in/is）
BAD_PATHS_TMPL='(process_status|risk_level|prediction_code|is_error|ProcessStatus|RiskLevel)[^}]*[=!][=]|[{][%][^}]*\b(if|elif)\b[^}]*(process_status|risk_level|prediction_code|is_error)'
BAD_PATHS_PY='\b(process_status|risk_level|prediction_code|is_error)\b\s*[=!][=]|\bif\s+.*\b(process_status|risk_level|prediction_code|is_error)\b'

echo "==== home.html 判断检查 ===="
if grep -n -E "$BAD_PATHS_TMPL" templates/home.html; then
    echo "FAIL: home.html 里有状态判断"
    FAIL=1
else
    echo "OK"
fi

echo ""
echo "==== batch.html 判断检查 ===="
if grep -n -E "$BAD_PATHS_TMPL" templates/batch.html; then
    echo "FAIL: batch.html 里有状态判断"
    FAIL=1
else
    echo "OK"
fi

echo ""
echo "==== Streamlit 渲染函数 判断检查 ===="
awk '/^def _render_model/,/^def /{print NR": "$0}' app_streamlit.py > /tmp/_render_model.txt
awk '/^def _render_validation/,/^def /{print NR": "$0}' app_streamlit.py > /tmp/_render_validation.txt
awk '/^def _render_single/,/^def /{print NR": "$0}' app_streamlit.py > /tmp/_render_single.txt
awk '/^def _render_batch/,/^def /{print NR": "$0}' app_streamlit.py > /tmp/_render_batch.txt
cat /tmp/_render_model.txt /tmp/_render_validation.txt /tmp/_render_single.txt /tmp/_render_batch.txt > /tmp/_render_all.txt

if grep -n -E "$BAD_PATHS_PY" /tmp/_render_all.txt; then
    echo "FAIL: Streamlit 渲染函数里有状态判断"
    FAIL=1
else
    echo "OK"
fi

echo ""
echo "==== home.html 不能访问原始字段 (raw status fields) ===="
# home.html 只应该消费 single_*、risk_summary、single_risk_contributors_display 等展示字段
# 下列原始字段在渲染前应该已经映射完毕
RAW_FIELDS='(\.process_status|\.prediction_code|\.risk_level\b|\.risk_explanation|\.is_error|\.error_reason\b)'
if grep -n -E "$RAW_FIELDS" templates/home.html; then
    echo "FAIL: home.html 访问了原始状态字段，应该改用展示专用字段"
    FAIL=1
else
    echo "OK"
fi

echo ""
echo "==== batch.html 不能访问行原始字段 (row.process_status 等) ===="
ROW_RAW_FIELDS='row\.(process_status|prediction(?!_badge|_label|_code)|risk_level\b|is_error|error_reason\b|risk_explanation)'
if grep -n -E "$ROW_RAW_FIELDS" templates/batch.html; then
    echo "FAIL: batch.html 行中访问了原始状态字段"
    FAIL=1
else
    echo "OK"
fi

echo ""
echo "==== Streamlit 渲染函数不能访问原始字段 ===="
if grep -n -E "$RAW_FIELDS|display\[.process_status.\]|display\[.prediction_code.\]|display\[.risk_level.\]|display\[.is_error.\]|display\[.error_reason.\]|display\[.risk_explanation.\]" /tmp/_render_all.txt; then
    echo "FAIL: Streamlit 渲染函数访问了原始状态字段"
    FAIL=1
else
    echo "OK"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== ALL CHECKS PASSED ==="
    exit 0
else
    echo "=== SOME CHECKS FAILED ==="
    exit 1
fi
