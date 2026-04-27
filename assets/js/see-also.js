(function () {
  'use strict';

  var path = window.location.pathname;
  if (path.indexOf('/wiki/') !== 0) return;

  var target = document.querySelector('.page__content');
  if (!target) return;

  fetch('/assets/see-also.json', { credentials: 'same-origin' })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (data) {
      var recs = data[path];
      if (!recs || recs.length === 0) return;
      render(recs);
    })
    .catch(function (err) {
      if (window.console && console.warn) console.warn('[see-also]', err);
    });

  function render(recs) {
    var panel = document.createElement('section');
    panel.className = 'sa-panel';
    panel.setAttribute('aria-label', 'Related articles');

    var h = document.createElement('h3');
    h.className = 'sa-heading';
    h.textContent = 'See also';
    panel.appendChild(h);

    var ul = document.createElement('ul');
    ul.className = 'sa-list';
    recs.forEach(function (r) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = r.url;
      a.textContent = r.title;
      li.appendChild(a);
      ul.appendChild(li);
    });
    panel.appendChild(ul);

    target.appendChild(panel);
  }
})();
