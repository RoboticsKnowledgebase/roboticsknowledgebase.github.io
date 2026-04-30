(function () {
  "use strict";

  if (typeof d3 === "undefined") {
    var loader = document.getElementById("kg-loading");
    if (loader) loader.innerHTML = '<p style="color:#c9d1d9">Failed to load visualization library. <a href="/wiki/" style="color:#58a6ff">Browse articles instead</a></p>';
    return;
  }

  var data = window.GRAPH_DATA;
  if (!data) {
    var loader2 = document.getElementById("kg-loading");
    if (loader2) loader2.innerHTML = '<p style="color:#c9d1d9">Graph data failed to load. <a href="/wiki/" style="color:#58a6ff">Browse articles instead</a></p>';
    return;
  }

  var prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function dur(ms) { return prefersReducedMotion ? 0 : ms; }
  var entryAnimationDone = prefersReducedMotion;

  var categoryMap = {};
  data.categories.forEach(function (c) { categoryMap[c.id] = c; });

  // Build nodes
  var nodes = [];
  data.categories.forEach(function (c) {
    nodes.push({
      id: c.id, label: c.label, type: "category",
      color: c.color, url: c.url, articleCount: 0, radius: 0
    });
  });
  data.articles.forEach(function (a) {
    var cat = categoryMap[a.category];
    nodes.push({
      id: a.id, label: a.label, type: "article",
      color: cat ? cat.color : "#888", url: a.url, category: a.category
    });
    var catNode = nodes.find(function (n) { return n.id === a.category; });
    if (catNode) catNode.articleCount++;
  });
  nodes.forEach(function (n) {
    if (n.type === "category") n.radius = Math.max(22, 10 + Math.sqrt(n.articleCount) * 8);
  });

  // Build links
  var links = [];
  data.articles.forEach(function (a) {
    links.push({ source: a.category, target: a.id, type: "membership" });
  });
  data.edges.forEach(function (e) {
    links.push({ source: e.source, target: e.target, type: "cross", label: e.label });
  });

  var nodeMap = {};
  nodes.forEach(function (n) { nodeMap[n.id] = n; });

  // Adjacency + edge labels for hover highlighting
  var adjacency = {};
  var edgeLabels = {};
  function addAdj(a, b) {
    if (!adjacency[a]) adjacency[a] = {};
    adjacency[a][b] = true;
  }
  function addEdgeLabel(a, b, label) {
    if (!edgeLabels[a]) edgeLabels[a] = {};
    edgeLabels[a][b] = label;
  }
  links.forEach(function (l) {
    var sid = typeof l.source === "object" ? l.source.id : l.source;
    var tid = typeof l.target === "object" ? l.target.id : l.target;
    addAdj(sid, tid);
    addAdj(tid, sid);
    if (l.label) {
      addEdgeLabel(sid, tid, l.label);
      addEdgeLabel(tid, sid, l.label);
    }
  });

  // ── DOM setup ──
  var container = document.getElementById("kg-container");
  var width = container.clientWidth;
  var height = container.clientHeight;

  var svg = d3.select("#kg-svg")
    .attr("width", width)
    .attr("height", height)
    .attr("role", "img")
    .attr("aria-label", "Interactive knowledge graph showing " + data.articles.length + " articles across " + data.categories.length + " categories");

  // Defs
  var defs = svg.append("defs");

  // Category glow filter only (removed article glow for performance)
  var catGlow = defs.append("filter").attr("id", "cat-glow")
    .attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
  catGlow.append("feGaussianBlur").attr("stdDeviation", "5").attr("result", "blur");
  catGlow.append("feComposite").attr("in", "SourceGraphic").attr("in2", "blur").attr("operator", "over");

  var g = svg.append("g").attr("class", "kg-graph");

  // Zoom
  var zoom = d3.zoom()
    .scaleExtent([0.15, 5])
    .on("zoom", function (event) {
      g.attr("transform", event.transform);
      currentZoom = event.transform.k;
      updateLabelVisibility();
    });
  svg.call(zoom);

  var currentZoom = 1;

  // Groups (z-order)
  var linkGroup = g.append("g").attr("class", "links");
  var nodeGroup = g.append("g").attr("class", "nodes");
  var labelGroup = g.append("g").attr("class", "labels");

  // ── Force simulation ──
  var simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(function (d) { return d.id; })
      .distance(function (d) {
        return d.type === "membership" ? 70 : 220;
      })
      .strength(function (d) {
        return d.type === "membership" ? 0.6 : 0.1;
      })
    )
    .force("charge", d3.forceManyBody()
      .strength(function (d) { return d.type === "category" ? -600 : -80; })
    )
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide()
      .radius(function (d) { return d.type === "category" ? d.radius + 8 : 10; })
    )
    .force("x", d3.forceX(width / 2).strength(0.02))
    .force("y", d3.forceY(height / 2).strength(0.02))
    .alphaDecay(0.018)
    .alphaMin(0.005)
    .velocityDecay(0.35);

  // ── Render links ──
  var linkElements = linkGroup.selectAll("line")
    .data(links)
    .enter().append("line")
    .attr("class", function (d) { return "kg-link kg-link-" + d.type; })
    .style("stroke", function (d) {
      if (d.type === "membership") {
        var cat = nodeMap[typeof d.source === "object" ? d.source.id : d.source];
        return cat ? cat.color : "#333";
      }
      return "#555";
    })
    .style("stroke-width", function (d) { return d.type === "membership" ? 0.5 : 1; })
    .style("stroke-opacity", function (d) { return d.type === "membership" ? 0.12 : 0.25; })
    .style("opacity", 0);

  // ── Render nodes ──
  var nodeElements = nodeGroup.selectAll("circle")
    .data(nodes)
    .enter().append("circle")
    .attr("class", function (d) { return "kg-node kg-node-" + d.type; })
    .attr("r", 0)
    .attr("tabindex", 0)
    .attr("role", "link")
    .attr("aria-label", function (d) {
      return d.label + " (" + (d.type === "category" ? d.articleCount + " articles" : (categoryMap[d.category] || {}).label || "") + ")";
    })
    .style("fill", function (d) { return d.color; })
    .style("fill-opacity", function (d) { return d.type === "category" ? 0.9 : 0.7; })
    .style("stroke", function (d) { return d.type === "category" ? "#fff" : "none"; })
    .style("stroke-width", function (d) { return d.type === "category" ? 1.5 : 0; })
    .style("stroke-opacity", 0.3)
    .style("cursor", "grab")
    .style("filter", function (d) { return d.type === "category" ? "url(#cat-glow)" : "none"; })
    .style("opacity", 0)
    .on("mouseover", handleMouseOver)
    .on("mousemove", handleMouseMove)
    .on("mouseout", handleMouseOut)
    .on("click", handleClick)
    .on("keydown", function (event, d) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handleClick(event, d);
      }
    })
    .on("focus", function (event, d) { handleMouseOver(event, d); })
    .on("blur", function () { handleMouseOut(); })
    .call(d3.drag()
      .on("start", dragStart)
      .on("drag", dragging)
      .on("end", dragEnd)
    );

  // ── Render labels ──
  var labelElements = labelGroup.selectAll("text")
    .data(nodes)
    .enter().append("text")
    .attr("class", function (d) { return "kg-label kg-label-" + d.type; })
    .text(function (d) { return d.label; })
    .style("fill", "#e0e0e0")
    .style("font-size", function (d) { return d.type === "category" ? "13px" : "9px"; })
    .style("font-weight", function (d) { return d.type === "category" ? "700" : "400"; })
    .style("text-anchor", "middle")
    .style("pointer-events", "none")
    .style("paint-order", "stroke")
    .style("stroke", "rgba(10,14,23,0.85)")
    .style("stroke-width", function (d) { return d.type === "category" ? "3px" : "2.5px"; })
    .style("opacity", 0)
    .style("user-select", "none");

  // ── Tick ──
  var tickedOnce = false;
  simulation.on("tick", function () {
    linkElements
      .attr("x1", function (d) { return d.source.x; })
      .attr("y1", function (d) { return d.source.y; })
      .attr("x2", function (d) { return d.target.x; })
      .attr("y2", function (d) { return d.target.y; });

    nodeElements
      .attr("cx", function (d) { return d.x; })
      .attr("cy", function (d) { return d.y; });

    // Only update article labels when they're visible
    labelElements.filter(function (d) { return d.type === "category"; })
      .attr("x", function (d) { return d.x; })
      .attr("y", function (d) { return d.y + d.radius + 16; });

    if (currentZoom > 1.3 || isHighlighting) {
      labelElements.filter(function (d) { return d.type === "article"; })
        .attr("x", function (d) { return d.x; })
        .attr("y", function (d) { return d.y + 14; });
    }

    if (!tickedOnce) {
      tickedOnce = true;
      var loader = document.getElementById("kg-loading");
      if (loader) setTimeout(function () { loader.classList.add("hidden"); }, 200);
    }
  });

  // Pause simulation when tab is hidden
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) simulation.stop();
    else simulation.alpha(0.1).restart();
  });

  // ── Entry animation ──
  if (prefersReducedMotion) {
    nodeElements.style("opacity", 1).attr("r", function (d) { return d.type === "category" ? d.radius : 5; });
    labelElements.style("opacity", function (d) { return d.type === "category" ? 1 : 0; });
    linkElements.style("opacity", 1);
  } else {
    setTimeout(function () {
      nodeElements.filter(function (d) { return d.type === "category"; })
        .transition().duration(800).ease(d3.easeQuadOut)
        .style("opacity", 1)
        .attr("r", function (d) { return d.radius; });
      labelElements.filter(function (d) { return d.type === "category"; })
        .transition().duration(800).ease(d3.easeQuadOut).style("opacity", 1);
    }, 300);

    setTimeout(function () {
      nodeElements.filter(function (d) { return d.type === "article"; })
        .transition().duration(1000).ease(d3.easeQuadOut)
        .style("opacity", 1)
        .attr("r", 5);
      linkElements.filter(function (d) { return d.type === "membership"; })
        .transition().duration(1000).ease(d3.easeQuadOut).style("opacity", 1);
    }, 800);

    setTimeout(function () {
      linkElements.filter(function (d) { return d.type === "cross"; })
        .transition().duration(1200).ease(d3.easeQuadOut).style("opacity", 1);
    }, 1500);

    setTimeout(function () {
      entryAnimationDone = true;
      updateLabelVisibility();
    }, 2800);
  }

  // ── Interactions ──
  var tooltip = document.getElementById("kg-tooltip");
  var isHighlighting = false;

  function positionTooltip(event) {
    var tw = tooltip.offsetWidth || 180;
    var th = tooltip.offsetHeight || 80;
    var cx = event.clientX || event.pageX;
    var cy = event.clientY || event.pageY;
    var tx = Math.min(cx + 16, window.innerWidth - tw - 12);
    var ty = Math.min(cy - 10, window.innerHeight - th - 12);
    tooltip.style.left = Math.max(8, tx) + "px";
    tooltip.style.top = Math.max(8, ty) + "px";
  }

  function buildTooltipContent(d) {
    var connected = adjacency[d.id] || {};
    var connCount = Object.keys(connected).length;
    var catLabel = d.type === "category" ? d.label : (categoryMap[d.category] || {}).label || "";

    var html =
      '<div class="kg-tooltip-title">' + escapeHtml(d.label) + '</div>' +
      '<div class="kg-tooltip-cat" style="color:' + d.color + '">' +
      (d.type === "category" ? d.articleCount + " articles" : escapeHtml(catLabel)) +
      '</div>' +
      '<div class="kg-tooltip-conn">' + connCount + ' connections</div>';

    // Show cross-edge labels
    var crossConns = [];
    var myEdgeLabels = edgeLabels[d.id] || {};
    Object.keys(myEdgeLabels).forEach(function (targetId) {
      var targetNode = nodeMap[targetId];
      if (targetNode && targetNode.type !== "category") {
        crossConns.push({ name: targetNode.label, rel: myEdgeLabels[targetId] });
      }
    });

    if (crossConns.length > 0) {
      html += '<div class="kg-tooltip-relations">';
      var shown = crossConns.slice(0, 5);
      shown.forEach(function (c) {
        html += '<div class="kg-tooltip-rel"><span class="kg-tooltip-rel-arrow">→</span> ' +
          escapeHtml(c.name) + ' <span class="kg-tooltip-rel-label">(' + escapeHtml(c.rel) + ')</span></div>';
      });
      if (crossConns.length > 5) {
        html += '<div class="kg-tooltip-rel kg-tooltip-rel-more">+' + (crossConns.length - 5) + ' more</div>';
      }
      html += '</div>';
    }

    return html;
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function handleMouseOver(event, d) {
    if (!entryAnimationDone) return;
    isHighlighting = true;
    var connected = adjacency[d.id] || {};

    nodeElements
      .transition().duration(dur(200)).ease(d3.easeQuadOut)
      .style("opacity", function (n) {
        if (n.id === d.id) return 1;
        if (connected[n.id]) return 1;
        return 0.08;
      })
      .attr("r", function (n) {
        if (n.id === d.id) {
          return n.type === "category" ? n.radius * 1.15 : 8;
        }
        return n.type === "category" ? n.radius : 5;
      });

    linkElements
      .transition().duration(dur(200)).ease(d3.easeQuadOut)
      .style("opacity", function (l) {
        var sid = l.source.id, tid = l.target.id;
        if (sid === d.id || tid === d.id) return 1;
        return 0.03;
      })
      .style("stroke-opacity", function (l) {
        var sid = l.source.id, tid = l.target.id;
        if (sid === d.id || tid === d.id) return 0.7;
        return 0.05;
      })
      .style("stroke-width", function (l) {
        var sid = l.source.id, tid = l.target.id;
        if (sid === d.id || tid === d.id) return l.type === "cross" ? 2.5 : 1.2;
        return l.type === "membership" ? 0.5 : 1;
      });

    labelElements
      .transition().duration(dur(200)).ease(d3.easeQuadOut)
      .style("opacity", function (n) {
        if (n.id === d.id) return 1;
        if (connected[n.id]) return 1;
        return 0.05;
      });

    tooltip.innerHTML = buildTooltipContent(d);
    tooltip.style.display = "block";
    positionTooltip(event);
  }

  function handleMouseMove(event) {
    if (tooltip.style.display === "block") positionTooltip(event);
  }

  function handleMouseOut() {
    if (!entryAnimationDone) return;
    isHighlighting = false;
    nodeElements
      .transition().duration(dur(250)).ease(d3.easeQuadOut)
      .style("opacity", function (n) {
        if (n.type === "category") return activeCategories[n.id] ? 1 : 0;
        return activeCategories[n.category] ? 1 : 0;
      })
      .attr("r", function (d) { return d.type === "category" ? d.radius : 5; });
    linkElements
      .transition().duration(dur(250)).ease(d3.easeQuadOut)
      .style("opacity", function (l) {
        var s = l.source, t = l.target;
        var sVis = s.type === "category" ? activeCategories[s.id] : activeCategories[s.category];
        var tVis = t.type === "category" ? activeCategories[t.id] : activeCategories[t.category];
        return (sVis && tVis) ? 1 : 0;
      })
      .style("stroke-opacity", function (d) { return d.type === "membership" ? 0.12 : 0.25; })
      .style("stroke-width", function (d) { return d.type === "membership" ? 0.5 : 1; });
    labelElements
      .transition().duration(dur(250)).ease(d3.easeQuadOut)
      .style("opacity", function (d) {
        if (d.type === "category") return activeCategories[d.id] ? 1 : 0;
        if (!activeCategories[d.category]) return 0;
        return computeArticleLabelOpacity();
      });
    tooltip.style.display = "none";
  }

  // Gradual label fade instead of hard threshold
  function computeArticleLabelOpacity() {
    if (currentZoom < 1.3) return 0;
    if (currentZoom > 2.2) return 0.9;
    return (currentZoom - 1.3) / (2.2 - 1.3) * 0.9;
  }

  var isDragging = false;

  function handleClick(event, d) {
    if (isDragging) return;
    if (d.url) window.open(d.url, "_self");
  }

  // Drag
  function dragStart(event, d) {
    isDragging = false;
    if (!event.active) simulation.alphaTarget(0.15).restart();
    d.fx = d.x;
    d.fy = d.y;
    d3.select(this).style("cursor", "grabbing");
  }
  function dragging(event, d) {
    isDragging = true;
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragEnd(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
    d3.select(this).style("cursor", "grab");
    setTimeout(function () { isDragging = false; }, 50);
  }

  // Label visibility based on zoom
  function updateLabelVisibility() {
    if (isHighlighting) return;
    var artOpacity = computeArticleLabelOpacity();
    labelElements.style("opacity", function (d) {
      if (d.type === "category") return 1;
      return artOpacity;
    });
  }

  // ── Search ──
  var searchInput = document.getElementById("kg-search");
  var searchClear = document.getElementById("kg-search-clear");
  var searchNoResults = document.getElementById("kg-search-no-results");

  searchInput.addEventListener("input", function () {
    var q = this.value.toLowerCase().trim();
    searchClear.style.display = q ? "block" : "none";
    if (!q) {
      handleMouseOut();
      if (searchNoResults) searchNoResults.style.display = "none";
      return;
    }
    var matches = {};
    var matchCount = 0;
    nodes.forEach(function (n) {
      if (n.label.toLowerCase().indexOf(q) !== -1) {
        matches[n.id] = true;
        matchCount++;
      }
    });

    if (searchNoResults) {
      searchNoResults.style.display = matchCount === 0 ? "block" : "none";
    }

    isHighlighting = true;
    nodeElements
      .transition().duration(dur(150))
      .style("opacity", function (n) { return matches[n.id] ? 1 : 0.06; });
    linkElements
      .transition().duration(dur(150))
      .style("opacity", 0.03);
    labelElements
      .transition().duration(dur(150))
      .style("opacity", function (n) { return matches[n.id] ? 1 : 0.03; });
  });

  searchInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var q = this.value.toLowerCase().trim();
      if (!q) return;
      var match = nodes.find(function (n) {
        return n.label.toLowerCase().indexOf(q) !== -1;
      });
      if (match && match.x != null) {
        svg.transition().duration(dur(750)).ease(d3.easeCubicOut).call(
          zoom.transform,
          d3.zoomIdentity.translate(width / 2, height / 2).scale(2.5).translate(-match.x, -match.y)
        );
      }
    }
    if (e.key === "Escape") {
      this.value = "";
      searchClear.style.display = "none";
      if (searchNoResults) searchNoResults.style.display = "none";
      handleMouseOut();
    }
  });

  searchClear.addEventListener("click", function () {
    searchInput.value = "";
    searchClear.style.display = "none";
    if (searchNoResults) searchNoResults.style.display = "none";
    handleMouseOut();
    searchInput.focus();
  });

  // ── Category Filters ──
  var filterContainer = document.getElementById("kg-filters");
  var activeCategories = {};
  data.categories.forEach(function (c) { activeCategories[c.id] = true; });

  data.categories.forEach(function (c) {
    var pill = document.createElement("button");
    pill.className = "kg-filter-pill active";
    pill.style.setProperty("--pill-color", c.color);
    pill.setAttribute("data-cat", c.id);
    pill.setAttribute("aria-pressed", "true");
    pill.textContent = c.label;
    pill.addEventListener("click", function () {
      activeCategories[c.id] = !activeCategories[c.id];
      this.classList.toggle("active", activeCategories[c.id]);
      this.setAttribute("aria-pressed", activeCategories[c.id] ? "true" : "false");
      applyFilters();
    });
    filterContainer.appendChild(pill);
  });

  function applyFilters() {
    nodeElements.style("display", function (n) {
      if (n.type === "category") return activeCategories[n.id] ? null : "none";
      return activeCategories[n.category] ? null : "none";
    });
    labelElements.style("display", function (n) {
      if (n.type === "category") return activeCategories[n.id] ? null : "none";
      return activeCategories[n.category] ? null : "none";
    });
    linkElements.style("display", function (l) {
      var s = l.source, t = l.target;
      var sVis = s.type === "category" ? activeCategories[s.id] : activeCategories[s.category];
      var tVis = t.type === "category" ? activeCategories[t.id] : activeCategories[t.category];
      return (sVis && tVis) ? null : "none";
    });
    simulation.alpha(0.3).restart();
  }

  // Reset filters
  document.getElementById("kg-reset-filters").addEventListener("click", function () {
    data.categories.forEach(function (c) { activeCategories[c.id] = true; });
    var pills = filterContainer.querySelectorAll(".kg-filter-pill");
    pills.forEach(function (p) {
      p.classList.add("active");
      p.setAttribute("aria-pressed", "true");
    });
    applyFilters();
    svg.transition().duration(dur(750)).ease(d3.easeCubicOut).call(
      zoom.transform,
      d3.zoomIdentity.translate(0, 0).scale(1)
    );
  });

  // ── Stats ──
  var articleCount = data.articles.length;
  var catCount = data.categories.length;
  var crossEdges = data.edges.length;
  var statsEl = document.getElementById("kg-stats");
  statsEl.setAttribute("role", "status");
  statsEl.innerHTML =
    '<span>' + catCount + ' categories</span><span class="kg-stat-sep">&bull;</span>' +
    '<span>' + articleCount + ' articles</span><span class="kg-stat-sep">&bull;</span>' +
    '<span>' + crossEdges + ' connections</span><span class="kg-stat-sep">&bull;</span>' +
    '<span class="kg-stat-hint">Click node to navigate &bull; Scroll to zoom &bull; Drag to pan</span>';

  // ── Resize (debounced) ──
  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      width = container.clientWidth;
      height = container.clientHeight;
      svg.attr("width", width).attr("height", height);
      simulation.force("center", d3.forceCenter(width / 2, height / 2));
      simulation.force("x", d3.forceX(width / 2).strength(0.02));
      simulation.force("y", d3.forceY(height / 2).strength(0.02));
      simulation.alpha(0.3).restart();
    }, 150);
  });

  // ── Legend ──
  var legendContent = document.getElementById("kg-legend-content");
  data.categories.forEach(function (c) {
    var item = document.createElement("div");
    item.className = "kg-legend-item";
    item.innerHTML = '<span class="kg-legend-dot" style="background:' + c.color + '"></span>' +
      '<span class="kg-legend-label">' + escapeHtml(c.label) + '</span>';
    legendContent.appendChild(item);
  });

  var legendToggle = document.getElementById("kg-legend-toggle");
  legendToggle.setAttribute("aria-expanded", "true");
  legendToggle.addEventListener("click", function () {
    var panel = document.getElementById("kg-legend");
    panel.classList.toggle("collapsed");
    var isCollapsed = panel.classList.contains("collapsed");
    this.textContent = isCollapsed ? "Legend" : "Hide";
    this.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
  });

  // Touch dismiss for tooltip
  svg.on("touchstart", function (event) {
    if (!event.target.classList || !event.target.classList.contains("kg-node")) {
      handleMouseOut();
    }
  }, { passive: true });
})();
