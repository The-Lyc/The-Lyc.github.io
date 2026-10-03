(() => {
  const layout = document.querySelector(".blog-layout");
  if (!layout) return;

  const panels = layout.querySelectorAll(".blog-nav-panel");
  const compactLayout = window.matchMedia("(max-width: 991px)");
  const updatePanels = () => panels.forEach((panel) => (panel.open = !compactLayout.matches));
  updatePanels();
  compactLayout.addEventListener("change", updatePanels);

  const toc = layout.querySelector("#blog-article-toc");
  const article = layout.querySelector(".blog-main-content");
  if (!toc || !article) return;

  const sections = Array.from(toc.querySelectorAll('a[href^="#"]')).flatMap((link) => {
    let id;
    try {
      id = decodeURIComponent(link.getAttribute("href").slice(1));
    } catch {
      return [];
    }
    const heading = document.getElementById(id);
    return heading && article.contains(heading) ? [{ link, heading }] : [];
  });
  if (!sections.length) return;

  let activeLink;
  let updatePending = false;
  const updateSection = () => {
    updatePending = false;
    const navbar = document.querySelector("#navbar");
    const scrollMargin = Number.parseFloat(window.getComputedStyle(sections[0].heading).scrollMarginTop) || 0;
    const offset = Math.max(navbar ? navbar.getBoundingClientRect().bottom + 24 : 24, scrollMargin) + 2;
    let current = sections[0];
    for (const section of sections) {
      if (section.heading.getBoundingClientRect().top > offset) break;
      current = section;
    }
    if (activeLink === current.link) return;

    activeLink?.removeAttribute("aria-current");
    activeLink = current.link;
    activeLink.setAttribute("aria-current", "location");

    if (!toc.closest("details")?.open) return;
    const bounds = toc.getBoundingClientRect();
    const linkBounds = activeLink.getBoundingClientRect();
    if (linkBounds.top < bounds.top) toc.scrollTop += linkBounds.top - bounds.top;
    else if (linkBounds.bottom > bounds.bottom) toc.scrollTop += linkBounds.bottom - bounds.bottom;
  };
  const scheduleUpdate = () => {
    if (updatePending) return;
    updatePending = true;
    window.requestAnimationFrame(updateSection);
  };

  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate, { passive: true });
  window.addEventListener("hashchange", scheduleUpdate);
  window.addEventListener("load", scheduleUpdate, { once: true });
  scheduleUpdate();
})();
