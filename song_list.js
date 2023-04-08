//song list
let All_song = [
   {
     name: "hey",
     path: "music/1.mp3",
     singer: "Null"
   },
   {
     name: "summer",
     path: "music/2.mp3",
     singer: "Null"
   },
   {
     name: "ukelele",
     path: "music/3.mp3",
     singer: "Null"
   },
   {
     name: "First Steps",
     path: "music/4.mp3",
     singer: "SoulProdMusic"
   },
   {
     name: "A Small Miracle",
     path: "music/5.mp3",
     singer: "Romarecord1973"
   },
   {
    name: "Coniferous forest",
    path: "music/6.mp3",
    singer: "orangery"
   },
   {
    name: "Risk",
    path: "music/6.mp3",
    singer: "StudioKolomna"
   },
   {
    name: "Smoke",
    path: "music/6.mp3",
    singer: "SoulProdMusic"
   },
   {
    name: "Waterfall",
    path: "music/6.mp3",
    singer: "RomanSenyMusic"
   },
];


/*tracks*/
let tracks = document.querySelector('.tracks');

//creating a list or generating Html
for (let i = 0; i < All_song.length; i++) {

  let Html = ` <div class="song">
      <div class="more">
      <audio src="${All_song[i].path}" id="music"></audio>
      <div class="song_info">
         <p id="title">${All_song[i].name}</p>
         <p>${All_song[i].singer}</p>
      </div>
      <button id="play_btn"><i class="fa fa-angle-right" aria-hidden="true"></i></button>
      </div>
    </div>`;

  tracks.insertAdjacentHTML("beforeend", Html);
};
