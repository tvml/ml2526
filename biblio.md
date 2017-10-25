---
layout: default
---

## Bibliografia 


<posts>
        <table>
            {% for post in site.categories.biblio %}
            <tr>
                <td><i class="icon-clock"></i> <time datetime="{{post.date"}}">{{post.date|date:"%d-%m-%Y"}}</time></td>  
                <td><a href="{{site.baseurl}}{{ post.url }}">{{ post.title }}</a></td>
            </tr>
            <tr>
                <td colspan='2'>
                {{ post.excerpt }}
                </td>
            </tr>
    	{% endfor %}
        </table>
</posts>


