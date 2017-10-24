---
layout: default
---

## Comunicazioni

<posts>
        <table>
            {% for post in site.posts %}
            <tr>
                <td><i class="icon-clock"></i> <time datetime="{{post.date"}}">{{post.date|date:"%d-%m-%Y"}}</time></td>  
                <td><a href="{{site.baseurl}}{{ post.url }}">{{ post.title }}</a></td>
                <td><span class="category"><i class="icon-tag"></i> {{post.categories | category_links}}</span></td>
            </tr>
            <tr>
                <td colspan='3'>
                {{ post.excerpt }}
                </td>
            </tr>
    	{% endfor %}
        </table>
</posts>



