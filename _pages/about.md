---
layout: about
title: about
permalink: /
subtitle: Ph.D. student · School of Computer Science and Engineering, UESTC

profile:
  align: right
  image: yichen-liu.jpg
  image_circular: false
selected_papers: false
social: false

announcements:
  enabled: false

latest_posts:
  enabled: false
---

<link rel="stylesheet" href="{{ '/assets/css/about.css' | relative_url | bust_file_cache }}">

I am **Yichen Liu (刘益辰)**, a Ph.D. student in the School of Computer Science and Engineering at the [University of Electronic Science and Technology of China (UESTC)](https://en.uestc.edu.cn/).

My research interests include **machine learning systems**, **GPU scheduling**, **real-time systems**, and **LLM inference serving**. My current work focuses on **runtime KV cache offloading**, **replay and recomputation control**, and **memory-constrained LLM inference**.

## education

**University of Electronic Science and Technology of China (UESTC)**

- **Ph.D. studies** · March 2026 – present (transferred from the master's program)
- **Master's studies** · 2024 – March 2026
- **Bachelor's degree in Computer Science and Technology** · September 2020 – June 2024

## publications

{% include bib_search.liquid %}

<div class="publications">

{% bibliography --file publications %}

</div>

## projects

<div class="about-projects">
  {% for project in site.data.cv.cv.sections.Projects %}
  {% assign project_id = project.name | slugify | prepend: 'project-' %}
  <article class="about-project-card" aria-labelledby="{{ project_id }}">
    <h3 class="about-project-name" id="{{ project_id }}">{{ project.name | escape }}</h3>
    <p class="about-project-topic">{{ project.topic | escape }}</p>
    <p class="about-project-description">{{ project.summary | escape }}</p>
    <a class="about-project-link" href="{{ project.url | escape }}">View on GitHub</a>
  </article>
  {% endfor %}
</div>

## awards

- **Second Place**, [EDAthon 2026](https://sites.google.com/view/ceda-hk/edathon-2026). Team: Yachen Wang and Yichen Liu.
- **First Prize**, [National College Student Operating System Competition (全国大学生操作系统能力大赛)](https://os.educg.net/#/oldDetail?name=2024%E5%B9%B4%E5%85%A8%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F%E8%83%BD%E5%8A%9B%E5%A4%A7%E8%B5%9B-%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E8%B5%9B%28%E5%85%A8%E5%9B%BD%29-OS%E5%8A%9F%E8%83%BD%E6%8C%91%E6%88%98%E8%B5%9B%E9%81%93), 2024.
- **Second Prize**, National College Student Information Security Competition — Project Track (全国大学生信息安全竞赛作品赛), 2024.
