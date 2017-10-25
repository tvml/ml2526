---
layout: default
---

## Lezioni svolte



<posts>
        <table>
            {% for post in site.categories.lezioni %}
            <tr>
                <td width="15%">{{post.data}}</td>  
                <td>{{ post.args }}</td>
            </tr>
    	{% endfor %}
        </table>
</posts>



