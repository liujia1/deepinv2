-- docx 版式：每个一级标题（章/前言/绪论/后记）前分页。
-- 说明：pandoc 的 docx writer 不认识 \newpage，必须直接注入 openxml 分页符。
-- 第一个一级标题前不分页（避免首页出现空白页）。
local first = true

function Header(el)
  if el.level == 1 then
    if first then
      first = false
      return nil
    end
    return {
      pandoc.RawBlock('openxml', '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'),
      el,
    }
  end
end
