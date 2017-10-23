---
layout: default
---

## Welcome to another page

_yay_

[back](./)

[Introduzione]({{ site.baseurl }}{%link slides/introduction.pdf %}).



Text can be **bold**, _italic_, or ~~strikethrough~~.

[Programma]({{ site.baseurl }}{%link programma.md %})

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# [](#header-1)Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## [](#header-2)Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### [](#header-3)Header 3

```
```python
# Python code with syntax highlighting.
# visualizzazione features di dataset iris
# sulla diagonale, distribuzione dei valori della feature nel training set
def scatter_matrix(df):
    features=[ x for x in df.columns if x!='class']
    classes=[ x for x in df['class'].unique()]
    nclasses=len(classes)
    nfeatures=len(features)
    data=np.array([np.array([np.array(df[df['class']==c][f]) for f in features]) for c in classes])
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    for i in range(nfeatures):
        flattened= np.array([val for sublist in data[:,i] for val in sublist])
        x = np.arange(min(flattened), max(flattened), .001)
        for j in range(nfeatures):
            ax = fig.add_subplot(nfeatures,nfeatures, i+nfeatures*j+1, axisbg="#F8F8F8")
            plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
            if i==j:
                y = st.gaussian_kde(flattened)(x)
                ax.plot(x, y, color=colors[3], alpha=.7)
                ax.fill_between(x, 0, y, color=colors[3], alpha=0.5)
                plt.xlim(min(flattened), max(flattened))
            else:
                for c in range(nclasses):
                    ax.scatter(data[c][i],data[c][j], color=colors[c],alpha=.9, s=10)
                plt.xlim(min(flattened), max(flattened))
            if i==0:
                ax.set_ylabel(features[j], fontsize=10)
            if j==nfeatures-1:
                ax.set_xlabel(features[i], fontsize=10)
    handles = [mpatches.Patch(color=colors[k], label=classes[k]) for k in range(nclasses)]
    plt.figlegend(handles, classes, 'upper center', ncol=nclasses, labelspacing=0. , fontsize=12)
    plt.show()

```

#### [](#header-4)Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### [](#header-5)Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### [](#header-6)Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
