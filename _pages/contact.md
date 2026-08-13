---
layout: page
permalink: /contact/
title: contact
description: get in touch
nav: true
nav_order: 6
---

<form action="https://formspree.io/f/xnpavkny" method="POST" class="col-md-8 mx-auto">
  <div class="mb-3">
    <label for="name" class="form-label">Name</label>
    <input type="text" name="name" id="name" class="form-control" required>
  </div>
  <div class="mb-3">
    <label for="email" class="form-label">Your email</label>
    <input type="email" name="_replyto" id="email" class="form-control" required>
  </div>
  <div class="mb-3">
    <label for="message" class="form-label">Message</label>
    <textarea name="message" id="message" class="form-control" rows="6" required></textarea>
  </div>
  <input type="text" name="_gotcha" style="display:none">
  <input type="hidden" name="_subject" value="New message from debajyotis.github.io">
  <button type="submit" class="btn btn-primary">Send</button>
</form>
