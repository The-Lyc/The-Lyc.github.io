---
layout: page
permalink: /blog/topics/
title: 技术笔记与专题
description: 按主题查找技术笔记，按章节阅读源码分析与课程笔记。
nav: false
lang: zh-CN
---

从系统架构到机器学习系统，这里按主题整理文章；源码阅读与课程笔记按章节顺序排列。

[查看全部博客文章]({{ '/blog/' | relative_url }})

<ul>
  {% for topic in site.data.blog_topics %}
    <li><a href="#{{ topic.id }}">{{ topic.title }}</a></li>
  {% endfor %}
</ul>

{% assign imported_posts = site.posts | where: 'notes_import', true %}
{% for topic in site.data.blog_topics %}
{% assign topic_posts = imported_posts | where_exp: 'post', 'post.categories contains topic.category' | sort: 'title' %}

  <section id="{{ topic.id }}">
    <h2>{{ topic.title }}</h2>
    <p>{{ topic.description }}</p>
    {% assign standalone_count = 0 %}
    {% for post in topic_posts %}
      {% unless post.series %}
        {% assign standalone_count = standalone_count | plus: 1 %}
      {% endunless %}
    {% endfor %}
    {% if standalone_count > 0 %}
      <ul>
        {% for post in topic_posts %}
          {% unless post.series %}
            <li><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></li>
          {% endunless %}
        {% endfor %}
      </ul>
    {% endif %}
    {% for series in topic.series %}
      {% assign series_posts = topic_posts | where: 'series', series.id | sort: 'series_order' %}
      {% if series_posts.size > 0 %}
        <h3 id="{{ series.id }}">{{ series.title }}</h3>
        <p>{{ series.description }}</p>
        <ol>
          {% for post in series_posts %}
            <li value="{{ post.series_order }}"><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></li>
          {% endfor %}
        </ol>
      {% endif %}
    {% endfor %}
  </section>
{% endfor %}
