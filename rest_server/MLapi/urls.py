from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns  
from MLapi import views

urlpatterns = [
    # url(r'^$', views.hexa_list),
    url(r'^$', views.hexa_test.as_view()),
   # url(r'^(?P<pk>[0-9]+)/$', views.hexa_detail),

#    path('hexa-test', )
]

urlpatterns = format_suffix_patterns(urlpatterns)