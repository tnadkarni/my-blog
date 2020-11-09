---
layout: post
title:  "Force Layout Graph using D3 in Jekyll" 
date:   2020-11-07 15:47:00
---
Here's an interactive force layout graph using D3 which allows us to position elements in a way that would be difficult to achieve using other means. Rather than simply going over the code and attaching an image of the result, I wanted to embed the D3 code into the Jekyll markdown file to make it interactive. 

<head>
<style>
	path.link {
	fill: none;
	stroke: #666;
	stroke-width: 1.5px;
	}
	.pin {
	stroke-width: 4px;
	}
	circle {
	fill: #ccc;
	stroke: #fff;
	stroke: black;
	stroke-width: 1.5px;
	}
	node.fixed == true {
	stroke-width: 3px;
	}
	text {
	fill: #000;
	font: 10px sans-serif;
	pointer-events: none;
	}
</style>
</head>
<div id="chart"></div>
<script>
// get the dataset
links = [
  {
    "source": "Harry",
    "target": "Sally",
    "value": 2.6
  },
  {
    "source": "Harry",
    "target": "Mario",
    "value": 2.5
  },
  {
    "source": "Sarah",
    "target": "Alice",
    "value": 0.2
  },
  {
    "source": "Eveie",
    "target": "Alice",
    "value": 0.5
  },
  {
    "source": "Peter",
    "target": "Alice",
    "value": 1.6
  },
  {
    "source": "Mario",
    "target": "Alice",
    "value": 0.4
  },
  {
    "source": "James",
    "target": "Alice",
    "value": 1.6
  },
  {
    "source": "Alice",
    "target": "James",
    "value": 1.1
  },
  {
    "source": "Harry",
    "target": "Carol",
    "value": 2.7
  },
  {
    "source": "Harry",
    "target": "Nicky",
    "value": 2.8
  },
  {
    "source": "Bobby",
    "target": "Frank",
    "value": 0.8
  },
  {
    "source": "Alice",
    "target": "Mario",
    "value": 0.7
  },
  {
    "source": "Harry",
    "target": "Lynne",
    "value": 2.4
  },
  {
    "source": "Sarah",
    "target": "James",
    "value": 0.9
  },
  {
    "source": "Roger",
    "target": "James",
    "value": 1.9
  },
  {
    "source": "Maddy",
    "target": "James",
    "value": 0.3
  },
  {
    "source": "Sonny",
    "target": "Roger",
    "value": 1.5
  },
  {
    "source": "Roger",
    "target": "Sonny",
    "value": 1.9
  },
  {
    "source": "James",
    "target": "Roger",
    "value": 1.5
  },
  {
    "source": "Alice",
    "target": "Peter",
    "value": 1.1
  },
  {
    "source": "Johan",
    "target": "Peter",
    "value": 1.6
  },
  {
    "source": "Alice",
    "target": "Eveie",
    "value": 0.5
  },
  {
    "source": "Harry",
    "target": "Eveie",
    "value": 2.9
  },
  {
    "source": "Eveie",
    "target": "Harry",
    "value": 2.1
  },
  {
    "source": "Henry",
    "target": "Mikey",
    "value": 0.4
  },
  {
    "source": "Elric",
    "target": "Mikey",
    "value": 0.6
  },
  {
    "source": "James",
    "target": "Sarah",
    "value": 0.5
  },
  {
    "source": "Alice",
    "target": "Sarah",
    "value": 0.6
  },
  {
    "source": "James",
    "target": "Maddy",
    "value": 0.5
  },
  {
    "source": "Peter",
    "target": "Johan",
    "value": 1.7
  },
  {
    "source": "Sonny",
    "target": "Johan",
    "value": 1.0
  },
  {
    "source": "Johan",
    "target": "Sonny",
    "value": 2.0
  },
  {
    "source": "Anna",
    "target": "Robert",
    "value": 0.6
  },
  {
    "source": "Anna",
    "target": "Brian",
    "value": 0.4
  },
  {
    "source": "Anna",
    "target": "Shang",
    "value": 0.3
  },
  {
    "source": "Anna",
    "target": "Nilaksh",
    "value": 0.1
  },
  {
    "source": "Anna",
    "target": "Andy",
    "value": 0.2
  },
  {
    "source": "Anna",
    "target": "Sam",
    "value": 0.6
  },
  {
    "source": "Anna",
    "target": "Paras",
    "value": 0.4
  },
  {
    "source": "Anna",
    "target": "Nathan",
    "value": 0.3
  },
  {
    "source": "Anna",
    "target": "Jenny",
    "value": 0.3
  },
  {
    "source": "Anna",
    "target": "Fred",
    "value": 0.3
  },
  {
    "source": "Jenny",
    "target": "Wendy",
    "value": 0.2
  },
  {
    "source": "Jenny",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Jenny",
    "target": "Bhanu",
    "value": 0.2
  },
  {
    "source": "Jenny",
    "target": "Kira",
    "value": 0.2
  },
  {
    "source": "Jenny",
    "target": "Kiran",
    "value": 0.2
  },
  {
    "source": "Jenny",
    "target": "Varun",
    "value": 0.2
  },
  {
    "source": "Wendy",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Wendy",
    "target": "Bhanu",
    "value": 0.2
  },
  {
    "source": "Wendy",
    "target": "Kiran",
    "value": 0.2
  },
  {
    "source": "Wendy",
    "target": "Varun",
    "value": 0.2
  },
  {
    "source": "Bhanu",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Bhanu",
    "target": "Kira",
    "value": 0.2
  },
  {
    "source": "Bhanu",
    "target": "Kiran",
    "value": 0.2
  },
  {
    "source": "Kira",
    "target": "Wendy",
    "value": 0.2
  },
  {
    "source": "Kira",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Kira",
    "target": "Kiran",
    "value": 0.2
  },
  {
    "source": "Kira",
    "target": "Wendy",
    "value": 0.2
  },
  {
    "source": "Kiran",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Kiran",
    "target": "Varun",
    "value": 0.2
  },
  {
    "source": "Varun",
    "target": "Fred",
    "value": 0.2
  },
  {
    "source": "Varun",
    "target": "Bhanu",
    "value": 0.2
  },
  {
    "source": "Varun",
    "target": "Kira",
    "value": 0.2
  },
  {
    "source": "Maddy",
    "target": "Rita",
    "value": 0.6
  },
  {
    "source": "Rita",
    "target": "Steve",
    "value": 0.6
  },
  {
    "source": "Steve",
    "target": "Rita",
    "value": 0.6
  },
  {
    "source": "Steve",
    "target": "Sean",
    "value": 0.6
  },
  {
    "source": "Sean",
    "target": "Preston",
    "value": 0.6
  },
  {
    "source": "Preston",
    "target": "Sean",
    "value": 0.6
  },
  {
    "source": "Sean",
    "target": "Rita",
    "value": 0.6
  }
];
var nodes = {};
// Compute the distinct nodes from the links.
links.forEach(function(link) {
    link.source = nodes[link.source] ||
        (nodes[link.source] = {name: link.source});
    link.target = nodes[link.target] ||
        (nodes[link.target] = {name: link.target});
});
var width = 800,
    height = 500,
    color = d3.scale.category20c();
var force = d3.layout.force()
    .nodes(d3.values(nodes))
    .links(links)
    .size([width, height])
    .linkDistance(60)
    .charge(-150)
    .on("tick", tick)
    .start();
// Set the range
var  v = d3.scale.linear().range([0, 100]);
// Scale the range of the data
v.domain([0, d3.max(links, function(d) { return d.value; })]);
var svg = d3.select("#chart").append("svg")
    .attr("width", width)
    .attr("height", height);
 var borderPath = svg.append("rect")
  .attr("x", 0)
  .attr("y", 0)
  .attr("height", height)
  .attr("width", width)
  .style("stroke", "black")
  .style("fill", "none")
  .style("stroke-width", 1);
// build the arrow.
svg.append("svg:defs").selectAll("marker")
    .data(["end"])      // Different link/path types can be defined here
  .enter().append("svg:marker")    // This section adds in the arrows
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 15)
    .attr("refY", -1.5)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5");
// add the links and the arrows
var path = svg.append("svg:g").selectAll("path")
    .data(force.links())
  .enter().append("svg:path")
    .attr("class", function(d) { return "link " + d.type; });
path.style("stroke", function(link){
  if (link.value < 1) {return "blue";}
  if (link.value >= 1 & link.value <= 2) {return "green";}
  if (link.value > 2) {return "red";}
});
// define the nodes
var node = svg.selectAll(".node")
    .data(force.nodes())
    .enter().append("g")
    .attr("class", "node")
    .call(force.drag)
    .on("dblclick", function(d){ 
        if(d.fixed == false) {
          d3.select(this).selectAll("circle")
          .classed("pin", true)
          .classed("fixed", d.fixed = true);
        }
        else {
          d3.select(this).selectAll("circle")
          .classed("pin", false)
          .classed("fixed", d.fixed = false);
        }
        });
// add the nodes
node.append("circle")
    .attr("r", function(d){
        return (links.filter(function(p) {
          return(p.source == d | p.target == d);
        })).length;
    });
var label = node.append("text")
            .text(function(d) { return d.name; })
            .attr("dx", 10);
// add the curvy lines
function tick() {
    path.attr("d", function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
        return "M" +
            d.source.x + "," +
            d.source.y + "A" +
            dr + "," + dr + " 0 0,1 " +
            d.target.x + "," +
            d.target.y;
    });
    node
        .attr("transform", function(d) {
		    return "translate(" + d.x + "," + d.y + ")"; });
};
</script>

To start with, add the following to your <strong>_layouts/post.html</strong> file within the header tag. This will add D3 support to the markdown. Make sure you do not have d3 source code location mentioned within the script tag of your actual code - this may create version conflicts. 

{% highlight html%}
  <script src="https://code.jquery.com/jquery-2.2.3.min.js"></script>
  <script src="https://d3js.org/d3.v3.min.js"></script>
  <script src="https://d3js.org/queue.v1.min.js"></script>
  <script src="https://d3js.org/topojson.v1.min.js"></script>
  <script src="https://code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
{% endhighlight%}

This is what the skeleton will look like. Note that we do not have a body tag. So if you append your SVG element to the body tag, your chart will not display. Instead create a <strong>div</strong> element before your script and append your SVG to this.
{% highlight html%}
<head>
<style>
  /*Any CSS styling goes here */
</style>
</head>
<div id=chart></div>
<script>
  //Add data here if not in separate file
  var svg = d3.select("#chart").append("svg")
    .attr("width", width)
    .attr("height", height);
</script>
{% endhighlight%}

I was having some trouble linking to an external file so in this case, the data is inside the javascript file - or script tag of html file - itself in a JSON format. This is a list of dictionaries where each dict has a source, target and value key.

<img src="{{ site.url }}{{ site.baseurl}}/assets/images/linksdata.png">

Get the distinct nodes by iterating over the links data. We have added a D3 force function with a charge which simulates electrostatic effects across all nodes - it is negative since we want a repulsing force. We also add a callback function <i>tick</i> for every time the simulation iterates or when user interacts with the chart. 

{% highlight javascript%}
var nodes = {};
links.forEach(function(link) {
    link.source = nodes[link.source] ||
        (nodes[link.source] = {name: link.source});
    link.target = nodes[link.target] ||
        (nodes[link.target] = {name: link.target});

var force = d3.layout.force()
    .nodes(d3.values(nodes))
    .links(links)
    .size([width, height])
    .linkDistance(60)
    .charge(-150)
    .on("tick", tick)
    .start();

});
{% endhighlight%}

Create the link paths by binding the data to the <i>path</i>> SVG element which is the generic element to define a shape. The seemingly arbitrary numbers for the d attribute are defining the curvy lines between our nodes.
The color of the link is set based on the 'value' field.

{% highlight javascript%}
svg.append("svg:defs").selectAll("marker")
    .data(["end"])      // Different link/path types can be defined here
  .enter().append("svg:marker")    // This section adds in the arrows
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 15)
    .attr("refY", -1.5)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5");
// add the links
var path = svg.append("svg:g").selectAll("path")
    .data(force.links())
  .enter().append("svg:path")
    .attr("class", function(d) { return "link " + d.type; });
path.style("stroke", function(link){
  if (link.value < 1) {return "blue";}
  if (link.value >= 1 & link.value <= 2) {return "green";}
  if (link.value > 2) {return "red";}
});
{% endhighlight%}

We define the D3 nodes and append to circle elements. We allow the nodes to be dragged by the user and also add a functionality to fix the nodes in place when double-clicked.

{% highlight javascript%}
// define the nodes
var node = svg.selectAll(".node")
    .data(force.nodes())
    .enter().append("g")
    .attr("class", "node")
    .call(force.drag)
    .on("dblclick", function(d){ 
        if(d.fixed == false) {
          d3.select(this).selectAll("circle")
          .classed("pin", true)
          .classed("fixed", d.fixed = true);
        }
        else {
          d3.select(this).selectAll("circle")
          .classed("pin", false)
          .classed("fixed", d.fixed = false);
        }
        });
// add the nodes
node.append("circle")
    .attr("r", function(d){
        return (links.filter(function(p) {
          return(p.source == d | p.target == d);
        })).length;
    });
var label = node.append("text")
            .text(function(d) { return d.name; })
            .attr("dx", 10);
{% endhighlight%}

Finally define the tick function - this should update cordinates of the node and text labels. dx and dy are relative positiions of text to the nodes.

{% highlight javascript%}
function tick() {
    path.attr("d", function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
        return "M" +
            d.source.x + "," +
            d.source.y + "A" +
            dr + "," + dr + " 0 0,1 " +
            d.target.x + "," +
            d.target.y;
    });
    node
        .attr("transform", function(d) {
		    return "translate(" + d.x + "," + d.y + ")"; });
};
{% endhighlight%}

We have used D3 to create an interactive force directed layout graph with additional customizations. Hopefully this has convinced you to harness the visualization power of D3 for your next analysis!

This programming assignment was submitted as coursework for <i>[CSE6242](http://poloclub.gatech.edu/cse6242/2016fall/) Data and Visual Analytics (Fall 2016), Georgia Tech College of Computing</i>. 