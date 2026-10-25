-- 中文排版：去掉全角左括号前的空格。
-- citeproc 的叙述式引用（author-in-text）固定输出 "作者 （年份）"，多出一个空格；
-- 按中文出版规范，全角括号前不留空格。本过滤器须放在 --citeproc 之后执行。
function Inlines(el)
  local out = pandoc.Inlines({})
  for i, x in ipairs(el) do
    local keep = true
    if x.t == 'Space' then
      local nxt = el[i + 1]
      if nxt and nxt.t == 'Str' and nxt.text:sub(1, 3) == '（' then
        keep = false
      end
    end
    if keep then out:insert(x) end
  end
  return out
end
