---
layout: page
permalink: /contact/
title: contact
description: get in touch
nav: true
nav_order: 6
---

<div class="contact-wrapper">
  <p class="contact-intro">Have a question, spotted an issue in one of the papers, or just want to say hi? Send a message below.</p>

  <form action="https://formspree.io/f/xnpavkny" method="POST" class="contact-form">
    <div class="form-group">
      <label for="name">Name</label>
      <input type="text" name="name" id="name" required>
    </div>
    <div class="form-group">
      <label for="email">Your email</label>
      <input type="email" name="_replyto" id="email" required>
    </div>
    <div class="form-group">
      <label for="message">Message</label>
      <textarea name="message" id="message" rows="6" required></textarea>
    </div>
    <input type="text" name="_gotcha" style="display:none">
    <input type="hidden" name="_subject" value="New message from debajyotis.github.io">
    <button type="submit">Send</button>
  </form>
</div>
