---
layout: post
title:  "Gentle intro to D3"
date:   2020-10-15 15:47:00
---

D3.js is a javascript visualization library that allows for creation of interactive and highly custom visualizations. Most people tend to dread it after being introduced due to a steep learning curve but for anyone interested in Data Viz with some background in Javascript, it is definitely worth picking up as it provides visual creative freedom like no other tool in this domain.

A bit of background - D3 allows binding of data to the DOM. DOM or Document Object Model is the tree like hierarchiel structure of the html page which acts as a programming interface between Javascript and the HTML document. The DOM is created by the browser when the page is loaded and Javscript interacts with DOM to add dynamic elements to the HTML page. D3 can bind data to different elements of the DOM.

Setting up your D3 environment is pretty simple - simply create a .html skeleton file using your preferred text editor e.g. [SublimeText](https://www.sublimetext.com/) and include the D3.js library to your HTML webpage using the <strong>script</strong> tag. D3.js is an open-source library and the source code of the library is freely available on it's website - [https://d3js.org](https://d3js.org)

{% highlight html%}
<html>
<meta charset="utf-8">
<script src="https://d3js.org/d3.v6.js"></script>
<body>
	<script type="text/javascript">
	</script>>
</body>>
</html>
{% endhighlight%}

Before you start drawing you need a drawing board - this is where SVG (scalable vector graphics) comes in. SVG is an XML format that gives the option to draw multiple shapes - adding more flexibility to our visualizations. Let's add the SVG element to our container (in this case, our <strong>body</strong> element). This happens within the script tag:

{% highlight html%}
<script type="text/javascript">

		var width = 500;
			height = 300;

		var svg = d3.select("body").append("svg")
		    .attr("width", width)
		    .attr("height", height)
		    .style("background-color", "blue");

</script>
{% endhighlight%}

To view the output, simply save your .html file and open it in your default browser. You should see a blue rectangle towards to top-left (origin) of your page. Open the <strong>Developer console > Elements</strong> tab to see the power of D3. The DOM representation of the body element looks like this-

{% highlight html%}
<body>
	<script type="text/javascript">
	//same script as above
	</script>
	<svg width="500" height="300" style="background-color:blue;"></svg>
</body>
{% endhighlight%}

We see that <i>d3.select("body").append("svg")</i> essentially selected the body tag and inserted an svg element with defined attributes (height, width, background-color). When we talk about D3 "binding data to DOM elements", this is exactly what we mean - we were able to bind numerical values of the width, height variables (data) to our svg element within our body element. 

Now let's add a graphic to our SVG-
{% highlight html%}
<script type="text/javascript">
		var width = 500;
			height = 300;		
		//blue svg canvas
		var svg = d3.select("body").append("svg")
		    .attr("width", width)
		    .attr("height", height)
		    .style("background-color", "blue");

		//adding a pink rectangle onto our blue canvas
		svg.append("rect")
		    .attr("width", width/2)
		    .attr("height", height/2)
		    .attr("fill", "pink");
	</script>
{% endhighlight%}

<strong>rect</strong> is a predefined SVG shape element to create a rectangle graphic. To reflect the changes, simply save your html file and refresh your web page. The final visual should look like this-
<img src="{{ site.url }}{{ site.baseurl}}/assets/images/svg_rect.png">
That's it for the gentle introduction. The power of D3 will be better understood in the next article which goes over the creation of a force layout graph. 