$(document).ready(function () {

    $('#example , header').vegas({
        delay: 7000,
        timer: false,
        shuffle: true,
        transition: 'fade',
        transitionDuration: 9000,
        animation: 'random',

        slides: [
            {
                src: 'resources/css/img/back-1.jpg'
            },
            {
                src: 'resources/css/img/back-2.jpg'
            },
            {
                src: 'resources/css/img/back-3.jpg'
            }

  ]
        
    });
    
    
   
    
});
