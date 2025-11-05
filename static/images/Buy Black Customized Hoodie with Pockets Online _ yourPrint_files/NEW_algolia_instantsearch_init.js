var instaSearch;
var instaSearchCat;

jQuery(document).ready(function($) {
    
var loadingTrendingCats = false;

var loadScripts = function(){

    univsersalCustomPlugins.loadAsyncScript('https://www.yourprint.in/wp-content/themes/flatsome-child/algolia_assets/NEW_instantsearch.js@4.56.10_dist_instantsearch.production.min.js?ver=6.4.3').then(function() {

    return univsersalCustomPlugins.loadAsyncScript('https://www.yourprint.in/wp-content/themes/flatsome-child/algolia_assets/typesense-instantsearch-adapter.min.js?ver=6.4.3');

    }).then(function(){

    return univsersalCustomPlugins.loadAsyncStyle('https://cdn.jsdelivr.net/npm/instantsearch.css@8.1.0/themes/satellite-min.css');

    }).then(function(){

        initSearch();

    }).catch(function(error) {
        
        console.log(error);

    });

}

//https://lkwdwrd.com/wp-template-js-templates-wp
var hitsTemplate = `

<script type="text/html" id="tmpl-instantsearch-hit">

<div class="col-inner">
  
   <div class="product-small box ">
      <# if ( data.images.product_image ) { var uncachedImage = data.images.product_image.url.replace("&cfcache=all", "&uncached");  #>
      <div class="box-image">
         <div class="image-none image-preloader">
            <a href="{{ data.permalink }}">
            <img style="aspect-ratio: 5/6; object-fit: cover;" width="247" height="296" src="{{ data.images.product_image.url }}" onerror="this.onerror=null;this.src='{{ uncachedImage }}';"> </a>
         </div>
         <div class="image-tools is-small top right show-on-hover"></div>
         <div class="image-tools is-small hide-for-small bottom left show-on-hover"></div>
         <div class="image-tools grid-tools text-center hide-for-small bottom hover-slide-in show-on-hover"></div>
      </div>
      <# } #>
      <div class="box-text box-text-products">
         <div class="title-wrapper">
            <p style="display:none;" class="category uppercase is-smaller no-text-overflow product-cat ">
               {{data.taxonomies.product_cat[0].replace("Customized ","").replace("Custom ","")}} 
            </p>
            <p class="name product-title woocommerce-loop-product__title"><a href="{{ data.permalink }}">{{{ data._highlightResult.post_title.value }}}</a></p>
         </div>
         <div class="price-wrapper-search">
            <# if ( data.rating ) { var rating_percent = (data.rating/5) * 100 #>
            <div class="star-rating star-rating--inline" role="img" aria-label="Rated {{ data.rating }} out of 5"><span style="width:{{rating_percent}}%">Rated <strong class="rating">4.67</strong> out of 5</span></div>
            <# } #>
            <# if ( data.price ) { #>
            <span style="display:block;" class="price"><span class="woocommerce-Price-amount amount"><bdi style="margin-right: 5px;"><span class="woocommerce-Price-currencySymbol">₹</span>{{ data.price }}</bdi><bdi style=" text-decoration: line-through; color: grey;font-size: 0.9em; "><span class="woocommerce-Price-currencySymbol">₹</span>{{ data.marked_price }}</bdi></span></span>
            <# var cbamt = data.price * 0.1 <= 200 ? Math.round(data.price * 0.1) : 200; #>
            <a href="{{ data.permalink }}">
               <div class="cboffer">Cashback of <span class="cbamt">₹{{cbamt}}</span></div>
            </a>
            <# } #>
            
            <# if ( data.express_shipping ) { #>
<a href=""{{ data.permalink }}">
<div class="express-ship"><img src="https://www.yourprint.in/wp-content/uploads/2024/04/2769339.png" width="25" style=" width: 25px; margin-right: 5px; border-bottom: none; ">Express Shipping</div>
</a>
             <# } #>
            
			<# if ( data.cat_color_swatch_html ) { #>
			{{{data.cat_color_swatch_html}}}
			<# } #>
         </div>
      </div>
   </div>
</div>

</script>

`;

var stopWords = ["a","am","an","and","as","at","by","c's","co","do","de","eg","et","for","he","hi","i","i'd","i'm","ie","if","in","inc","is","it","its","me","my","nd","no","non","nor","not","of","off","oh","ok","on","or","per","que","qv","rd","re","so","sub","t's","th","the","to","too","two","un","up","us","vs","we"];
    
var searchBoxEl;

////Helper to get and set recent searches from browser storage
var getsetRecentSearches = function( recentSearch = "" ){

var totalSearchesToStore = 10;    
    
if ( !univsersalCustomPlugins.isLocalStorageAvb() ) return false;

var searches = localStorage.getItem("yp_recent_searches") ?? '[]';

searches = !univsersalCustomPlugins.isJson(searches) ? '[]' : searches;

searches = JSON.parse(searches);

if ( !recentSearch || recentSearch == "*" ) return searches;

if ( searches.includes(recentSearch) ) return searches;

if (searches.length >= totalSearchesToStore) searches = searches.slice(1, totalSearchesToStore);

searches.push(recentSearch);

localStorage.setItem("yp_recent_searches",JSON.stringify(searches));

return searches;
    
}

////Helper to create recent searches section
var createRecentSearches = function(){

var searches = getsetRecentSearches();

if ( !searches || !searches.length ) {
    
    $('.recent-searches .heading').hide();
    $('.recent-searches .content').html('');
    return;

}

var html = '';

$.each(searches,function(_,searchTerm){
    
html += `<span class="term">${searchTerm}</span>`;  

});

html += `<span style="color:red;text-decoration: none;" class="term clear">Clear History</span>`; 

$('.recent-searches .heading').show();
$('.recent-searches .content').html(html);

$('.recent-searches .content .term:not(.clear)').click(function(){
    searchBoxEl.val($(this).text()).trigger('input').focus();
});

$('.recent-searches .content .term.clear').click(function(){
    localStorage.removeItem("yp_recent_searches");
    toggleSearchInterface(null,true);
});
    
}

var createTrendingCategories = async function(){

if ( $('#ais-main .trending-categories .img-slider-horizontal').length || loadingTrendingCats ) return;

loadingTrendingCats = true;

var html = await univsersalCustomPlugins.makeAjaxRequest({action: 'get_trending_cats_for_search'});

loadingTrendingCats = false;

if ( !html ) {
    $('#ais-main .trending-categories').hide();
    return;
}

$('#ais-main .trending-categories .content').html(html);

$('#ais-main .trending-categories .exit').click(function(){
    toggleSearchInterface(false);
})
    
}


var queryWithoutStopWords = function(query) {
    
  const words = query.replace(/[&\/\\#,+()$~%.'":*?<>{}]/g, '').split(' ');
  return words
    .map(word => {
      if (stopWords.includes(word.toLowerCase())) {
        return null;
      } else {
        return word;
      }
    })
    .filter(w => w)
    .join(' ')
    .trim();
    
}

var insertHtml = function(){

searchBoxEl = $('.header-wrapper form.searchform [id*=woocommerce-product-search-field-]:visible');

if ( !searchBoxEl ) return;

//Disable form submit and replacing button element with div element to prevent button click on pressing enter key (as per default browser behavior most probably)
searchBoxEl.closest('form').submit(function(e){
        e.preventDefault();
}).find('button.ux-search-submit').replaceWith('<div class="ux-search-submit submit-button secondary button  icon mb-0" style="visibility: hidden;"> <i class="icon-cross"></i> </div>');


$('form.searchform .ux-search-submit').click(function(e){

e.preventDefault();

if ( !searchBoxEl.val() ) {
    toggleSearchInterface(false);
    } else { 
        searchBoxEl.val('').trigger('input').focus();
        //instaSearch.helper.setQuery('').search();
        toggleSearchInterface(null,true);
    }

});

$('body').append(hitsTemplate);

//Adding CSS
$('body').append(`<style>

#algolia-hits-categories .ais-Hits-list {
    display: flex;
    overflow-x: auto;
}

li.ais-Pagination-item--previousPage .ais-Pagination-link{
    margin-left: inherit !important;
}

.ais-Pagination-item--firstPage, .ais-Pagination-item--lastPage{
    display: none;
}


div#algolia-hits-categories .ais-Hits-item{
    padding: 0.5em;
    display: inline-flex;
}


@media screen and (min-width: 850px) {

header#header{
    position: sticky;
    top: 0;
}

body {
    overflow-x: initial !important;
}


}

@media screen and (max-width: 48em) {

.sticky-header-mobile-temp{
    position: sticky;
    top: 0;
}

.sticky-header-mobile-temp-body{
    overflow-x: initial !important;
}

}
  
.recent-searches .content .term {
    margin: 5px;
    display: inline-flex;
    padding: 3px;
    opacity: 0.8;
    font-size: 0.8em;
    text-decoration: underline;
    cursor: pointer;
}

#ais-main .img-slider-horizontal {
    display: contents !important;
}

#ais-main .img-slider-horizontal a {
    display: inline-flex !important;
    margin: 3px;
}

#ais-main .img-slider-horizontal p{
    font-size: 0.7em !important;
}
  
.ais-Hits-item{
    box-shadow:none;
}

.ais-Panel:has(.ais-Pagination){
    overflow-x: auto;
}

.searchform .ux-search-submit i {
    color: #808080ad;
    font-size: 1em !important;
}

.ais-Hits.ais-Hits--empty {
    text-align: center;
}

#algolia-stats .ais-Panel {
    margin: 5px 0 0 0;
}

.ux-search-submit.circle i {
    border-radius: 50%;
    border: 1px solid;
    padding: 2px 3px;
}

.ais-SearchBox-form {
margin-bottom: 0;
border-radius: 5px;
position:relative;
}

#algolia-search-box{
    padding-top: 10px;
    padding-bottom: 10px;
    background: white;
}

.ais-SearchBox-input{
    border-radius: 5px !important;
    padding-left:35px !important;
}

.ais-SearchBox-form::before{
    display:none;
}

.ais-SearchBox-input:focus{
    border-color: orange;
}

.ais-Pagination{
    display: contents;
}


.ais-ClearRefinements-button--disabled{
    visibility: hidden;
}

.ais-ClearRefinements-button{
    margin: 0;
    border:none;
    text-transform: none;
    box-shadow:none;
}

.ais-facets{
    padding: 10px;
}

 
  
  aside#ais-facets {
    padding-top: 0 !important
  }

  #ais-wrapper {
    display: none;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgb(255 255 255);
    z-index: 1000000;
    max-width: 1080px;
    margin-left: auto;
    margin-right: auto;
    background: white;
    min-height: 100vh;
  }

  input.ais-search-box--input {
    margin-top: 10px;
    background: white !important;
    margin-bottom: 10px;
    border-radius: 5px;
    font-size: 14px !important;
    padding: 10px 0 10px 40px !important;
    border: none !important
  }

  #algolia-search-box .search-icon {
    left: 15px !important;
    top: 14px !important;
    fill: #d1dcdf !important
  }

  #ais-facets {
    width: 100% !important;
    height: 100% !important
  }

  span.ais-refinement-list--count {
    display: none
  }

  h3.widgettitle {
    color: #f6861f
  }

  span.ais-search-box--reset-wrapper svg {
    width: 16px
  }

  span.ais-search-box--reset-wrapper {
    width: 25px;
    position: absolute;
    right: 70px !important;
    top: 10px !important;
    fill: rgb(177 188 191) !important
  }

  .ais-stats {
    top: 83px !important;
    font-size: 12px !important;
    background: white;
    padding-left: 20px
  }

  .ais-pagination {
    background: #FFF;
    box-shadow: 0 1px 1px 0 rgba(85, 95, 110, 0.2);
    border: solid 1px #D4D8E3;
    border-radius: 4px;
    display: inline-block;
    padding: 8px 16px;
    width: auto
  }

  .ais-pagination--item {
    border-radius: 4px;
    font-size: 14px;
    text-align: center;
    width: 28px
  }

  .ais-pagination--item__disabled {
    color: #BBB;
    opacity: .5;
    pointer-events: none;
    visibility: visible
  }

  .ais-pagination--item__active {
    background: #446084
  }

  .ais-pagination--item__active .ais-pagination--link {
    color: #FFF
  }

  .ais-hits--item {
    border: 0 !important
  }

  li.ais-pagination--item {
    margin-bottom: 0
  }

  
</style>`);


//Adding searchBox container
//$('<div id="algolia-search-box" style="display:none; vertical-align: middle;margin-bottom: 0.2em;padding-left: 10px;padding-right: 10px;"></div>').insertAfter($('form.searchform #woocommerce-product-search-field-0').is(':visible') ? 'form.searchform #woocommerce-product-search-field-0' : 'form.searchform #woocommerce-product-search-field-1' );


//Adding results section 
$(`<div id="ais-wrapper">
    <main id="ais-main" style=" padding: 0;width:100% ">
      <div id="algolia-stats" style="padding-left: 10px; font-size: 0.7em;padding-bottom:10px;border-bottom: 5px solid #f1f3f6;"></div>
      
       <div id="algolia-hits-categories" style="padding-left: 5px;padding-right: 5px; border-bottom: 5px solid #f1f3f6;"></div>
      
      <div id="algolia-hits" style="padding-top: 5px;padding-left: 5px;padding-right: 5px;position:relative;"></div>
      
        <div class="recent-searches" style="
        text-align: center;
        margin-top: 20px;
        ">
        
        <div class="heading" style="font-weight: bold;">Recent Searches</div>
        <div class="content"></div>
        
        </div>
        
        <div class="trending-categories" style="
        text-align: center;
        margin-top: 20px;
        ">
        
        <div class="heading" style="font-weight: bold;">Trending Categories</div>
        <div class="content" style="margin-top: 5px;"></div>
        
        <div class="exit" style=" color: red; margin-top: 10px; cursor: pointer; opacity: 0.8; font-size: 0.8em; ">Exit Search</div>
        
        </div>
    
        
      <div id="algolia-pagination" style=" text-align: center;margin-left: 10px; margin-right: 10px;display: flex; justify-content: center;"></div>
    </main>
    <aside id="ais-facets" style="display: none;">
      <section class="ais-facets" id="facet-confirm" style="background: white; width: 100%;left: 0;box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);display:flex; align-items: center; margin-bottom: 15px;">
        
          <div style=" text-align: center; width: 33%; float: left; padding:10px;">
            <svg class="close-filter" xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </div>
          <div style="width: 33%; text-align: center; float: left;padding:10px;" id="facet-clear"></div>
          <div style=" width: 34%; text-align: center; float: left;padding:10px; ">Apply</div>
       
      </section>
      <section>
      <section class="ais-facets" id="facet-categories" style="margin-bottom: 10px;"></section>
      </section>
      <section class="ais-facets" id="facet-price" style="margin-top: 55px;"></section>
    </aside>
  </div>`).insertAfter('#header');
  




}

var toggleSearchInterface = function( toggle = true, toggleSectionsOnly = false ){
    
    $('#algolia-hits').removeClass('loading-card');
    
    if ( !toggleSectionsOnly ) {
    if ( toggle ) {
        //$('body').addClass('no-scroll-bar');
        $('#ais-wrapper').css('display', 'flex');
        $('form.searchform .ux-search-submit').css('visibility','visible');
        $('main#main,.footer,.shop-page-title,.breadcrumbs-section').addClass('hidden');
        
        $('body').addClass('sticky-header-mobile-temp-body');
        $('header#header').addClass('sticky-header-mobile-temp');
        
    } else {
        //$('body').removeClass('no-scroll-bar');
        $('#ais-wrapper').css('display', 'none');
        $('form.searchform .ux-search-submit').css('visibility','hidden');
        $('main#main,.footer,.shop-page-title,.breadcrumbs-section').removeClass('hidden');
        
        $('body').removeClass('sticky-header-mobile-temp-body');
        $('header#header').removeClass('sticky-header-mobile-temp');
        
        document.body.scrollTop = document.documentElement.scrollTop = 0;
        
        if ( typeof fancyProductDesigner == "object" && fancyProductDesigner.currentViewInstance ) fancyProductDesigner.currentViewInstance.resetCanvasSize();
        
    }
    }
    
    if ( !searchBoxEl.val() ) {
        $('#algolia-stats, #algolia-hits, #algolia-hits-categories, #algolia-pagination').hide();
        $('.recent-searches, .trending-categories').show();
        
        $('form.searchform .ux-search-submit').addClass('circle');
        
    } else {
        $('#algolia-stats, #algolia-hits, #algolia-hits-categories, #algolia-pagination').show();
        $('.recent-searches, .trending-categories').hide();
        
        $('form.searchform .ux-search-submit').removeClass('circle');
    }
    
    if ( typeof instaSearch == "object" && instaSearch.helper.lastResults ) {
        
    !instaSearch.helper.lastResults.nbHits || !instaSearch.helper.getQuery().query ? $('.recent-searches, .trending-categories').show() : $('.recent-searches, .trending-categories').hide();
        
    }
    
    createRecentSearches();
    createTrendingCategories();
    
}

var initSearch = function() {
	
	//Checking if instantsearch is defined, although scripts are in order, with wp-rocket delay js, instantsearch is undefined error is coming, added this check to fix the error. Error is coming on product pages only
	if( typeof instantsearch == "undefined") {
	    setTimeout(initSearch, 300);
	    return;
	}

	insertHtml();
	
	if ( !searchBoxEl ) return;

    var searchBarElement;
	
	var keyupTimer = null;
	
	var searchEnterStopped = false;
	
	var searchTimer;


/***** Typesense adapter ****************************/

const typesenseInstantsearchAdapter = new TypesenseInstantSearchAdapter({
  server: {
    apiKey: "5CWxDCV2Fu3OyrNptg2PEIwH2duIneAD", // Be sure to use an API key that only allows search operations
    nodes: [
      {
        host: "fehrwvanbd3ly4zgp-1.a1.typesense.net",
        path: "", // Optional. Example: If you have your typesense mounted in localhost:8108/typesense, path should be equal to '/typesense'
        port: "443",
        protocol: "https",
      },
    ],
    cacheSearchResultsForSeconds: 2 * 60, // Cache search results from server. Defaults to 2 minutes. Set to 0 to disable caching.
  },
  // The following parameters are directly passed to Typesense's search API endpoint.
  //  So you can pass any parameters supported by the search endpoint below.
  //  query_by is required.
  /**additionalSearchParameters: {
    query_by: "taxonomies.product_cat,post_title,taxonomies,taxonomies_hierarchical",
    //text_match_type: "max_weight",
   // query_by_weights: "2,1",
    sort_by:"menu_order:asc,total_sales:desc,_text_match:desc", //_text_match(buckets:1000):desc,
    per_page: 52,
    prioritize_exact_match: false
  },**/
  collectionSpecificSearchParameters:{
      "products_live": {
          //post_title,taxonomies,taxonomies_hierarchical,content
    query_by: "pseudo_title,taxonomies.product_cat,taxonomies,taxonomies_hierarchical,content",
    //query_by_weights: "3,3,3,3,1",
    //text_match_type: "max_weight",
    prioritize_exact_match: false,
    prioritize_token_position: false,
   // sort_by:"menu_order:asc,_text_match:desc,total_sales:desc",
     prioritize_num_matching_fields: false,
  //menu_order:asc,
   sort_by:"_text_match:desc,total_sales:desc",
    per_page: 40,
      },
    "categories_live": {
        query_by: "pseudo_title",
        sort_by: "_text_match:desc,cat_id:asc",
        per_page: 10
    }  
      
  }
});
const tsSearchClient = typesenseInstantsearchAdapter.searchClient;

/*************************************************/

//Category search
instaSearchCat = instantsearch({
        searchClient: tsSearchClient,
        indexName: 'categories_live'
        });
        
instaSearchCat.addWidget(
        instantsearch.widgets.hits({
            container: '#algolia-hits-categories',
            templates: {
                //item: wp.template('instantsearch-hit-categories'),
                item(hit, { html, components }) {
                
                var catName = hit.title.replace("Customized ","").replace("Custom ","");
                var catImage = hit.image ? 'https://www.yourprint.in/new-admin-ajax.php?action=resize_outer_image&url=' + hit.image + '&width=200&cfcache=all' : "";
                
                var uncachedImage = catImage.replace("&cfcache=all", "&uncached"); //Wanted to put this in onerror att for image but it doesn't show on front end probably due to some issue in instantsearch.js, onerror when used inside <script type="text/html" id="tmpl-instantsearch-hit"> shows up
                
                return html`
                <a href="${hit.link}">
                
                <div style="width:75px;flex-shrink: 0;margin-right: 10px;position: relative;">
            
                <img src="${catImage}" width="75" height="90" style="border-radius: 10px;" loading="lazy" />
              
                
                <div style="position: absolute; bottom: 2px; width: auto; min-width: 50%; background: rgb(255 255 255 / 90%); padding: 4px; left: 50%; transform: translateX(-50%); text-align: center; border-radius: 5px;line-height: 1; overflow: hidden; font-size: 9px; white-space: initial; text-overflow: ellipsis; font-weight: bold;max-width: 120%;width: 100%; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">${catName}</div>
             
                </div>
                
                </a>
                `;
                
                },
                empty(results, { html }) {
                return html``;
                 }
            },
        })
        );
               
instaSearchCat.start();

/**************************************/

//Product search
instaSearch = instantsearch({
        searchClient: tsSearchClient,
        indexName: 'products_live',
        routing: true,
        searchParameters: {
            facetingAfterDistinct: true,
            highlightPreTag: '__ais-highlight__',
            highlightPostTag: '__/ais-highlight__',
            hitsPerPage: 52,
        }
        });
        


    /****** Search box widget
    instaSearch.addWidget(
        instantsearch.widgets.searchBox({
            container: '#algolia-search-box',
            reset: true,
            magnifier: false,
            placeholder: 'Search for Products...',
            wrapInput: false,
			searchOnEnterKeyPressOnly:false,
			queryHook: function(inputValue, search) {
			
			inputValue = queryWithoutStopWords(inputValue);
			
			console.log(inputValue);
				
		    clearTimeout(searchTimer);
			
		    searchTimer = setTimeout(function(){ search(inputValue); }, 1000);
		
			},

        })
    );
    ********/

    /* Stats widget */
    
    const statsWidget = instantsearch.widgets.panel({
     hidden: function(results){
      //console.log(results);
      return !results.nbHits;
     },
        })(instantsearch.widgets.stats);
    
    
    
    instaSearch.addWidget(
        statsWidget({
            container: '#algolia-stats',
            templates: {
                text(data, { html }){
                    
            let count = '';

            if (data.hasManyResults) {
             count += `${data.nbHits} results`;
            } else if (data.hasOneResult) {
             count += `1 result`;
             } else {
             count += `no result`;
            }
                    
            return html`<span>Showing ${count} for <b>${data.query}</b></span>`;
                    
                }
            },


        })
    );

    /* Hits widget */
    instaSearch.addWidget(
        instantsearch.widgets.hits({
            container: '#algolia-hits',

            templates: {
                empty: 'No results were found for "<strong>{{query}}</strong>".',
                item: wp.template('instantsearch-hit')
            },

            cssClasses: {
                list: 'products row row-small large-columns-5 medium-columns-4 small-columns-2',
                item: ['product-small col has-hover product type-product status-publish instock has-post-thumbnail purchasable '],
            }
        })
    );
    
    const pagination = instantsearch.widgets.panel({
     hidden: function(results){
      //console.log(results);
      return results.nbPages === 1 ||  results.nbPages === 0;
     },
        })(instantsearch.widgets.pagination);

    /* Pagination widget */
    instaSearch.addWidget(
        pagination({
            container: '#algolia-pagination',
           // scrollTo: '#algolia-stats'
        })
    );


    /*** Refinement widget 

    var categoriesFilterWIdget = instantsearch.widgets.panel({
        templates:{
            header: "Categories"
        },
        })(instantsearch.widgets.hierarchicalMenu);
        
        ****/

     /***
    instaSearch.addWidget(
       categoriesFilterWIdget({
            container: '#facet-categories',
            attributes: ['taxonomies_hierarchical.product_cat.lvl0','taxonomies_hierarchical.product_cat.lvl1','taxonomies_hierarchical.product_cat.lvl2'],
            operator: 'or',
            limit: 10,
            sortBy: ['isRefined:desc', 'count:desc', 'name:asc']
        })
    );
    ***/
    
    /*** Price filter
     var priceFilterWIdget = instantsearch.widgets.panel({
        templates:{
            header: "Price"
        },
        })(instantsearch.widgets.rangeSlider);
        
        
     instaSearch.addWidget(
       priceFilterWIdget({
            container: '#facet-price',
            attribute: 'price'
        })
    ); 
    ****/
    

    /*** Clear all refinements widget 
    instaSearch.addWidget(
        instantsearch.widgets.clearRefinements({
            container: '#facet-clear',
            templates:{
                resetLabel: 'Reset'
            }
        })
    ); **/

instaSearch.start();

if ( $('.header-wrapper form.searchform [id*=woocommerce-product-search-field-]').val() ){
    
    var tempInp = queryWithoutStopWords($('.header-wrapper form.searchform [id*=woocommerce-product-search-field-]').val());
    
    instaSearch.helper.setQuery(tempInp).search();
    instaSearchCat.helper.setQuery(tempInp).search();
    toggleSearchInterface(true);
    
}

    
instaSearch.on('render', function(event) {
   
  toggleSearchInterface(null,true);
   
  // instaSearch.helper.lastResults ? console.log(instaSearch.helper.lastResults.nbHits) : null;
   
   if ( instaSearch.helper.lastResults && instaSearch.helper.lastResults.nbHits ) getsetRecentSearches( instaSearch.helper.lastResults.query );
   
   if ( $('#ais-wrapper').is(':visible') ) document.body.scrollTop = document.documentElement.scrollTop = 0;
   
});
    
$('.header-wrapper form.searchform [id*=woocommerce-product-search-field-]').on('focus',function(){
//keypress keydown keyup 
toggleSearchInterface(true);
  
}).on('input',function(){
    
var inputValue = queryWithoutStopWords($(this).val());

$('#algolia-hits').addClass('loading-card');
				
clearTimeout(searchTimer);

searchTimer = setTimeout(function(){ 
    instaSearch.helper.setQuery(inputValue).search();
    instaSearchCat.helper.setQuery(inputValue).search();
    toggleSearchInterface(true);
}, 500);
		
    
});

}

var timer = setTimeout(loadScripts,9000);

$('.header-wrapper form.searchform [id*=woocommerce-product-search-field-]').on('focus',function(){

if ( typeof instantsearch != "undefined" ) return;

clearTimeout(timer);

loadScripts();

});


});